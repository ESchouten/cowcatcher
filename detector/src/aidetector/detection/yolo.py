import logging
import os
import pathlib
from datetime import datetime
from typing import Any

from aidetector.utils.config import (
    Confidence,
    Crop,
    Detection,
    ImageSet,
    OnnxConfig,
    YoloConfig,
    min_confidence,
)
from aidetector.utils.onnx import should_half, should_rect
from aidetector.utils.version import TYPE
from numpy import ndarray
from ultralytics import YOLO

logger = logging.getLogger(__name__)


def _patch_windows_path_checkpoints() -> None:
    if os.name != "nt":
        pathlib.WindowsPath = pathlib.PosixPath


class YoloResultMapper:
    def __init__(self, class_confidences: dict[int, tuple[str, float]]):
        self.class_confidences = class_confidences

    def detections_from_result(
        self, result: Any, frames: list[tuple[datetime, ndarray]]
    ) -> list[Detection] | None:
        crops, confidences = self._collect_crops(result)
        if not crops:
            return None

        detections = [
            Detection(frame_date, ImageSet(frame, list(crops)), {})
            for frame_date, frame in frames[:-1]
        ]
        detections.append(
            Detection(
                frames[-1][0],
                ImageSet(frames[-1][1], list(crops)),
                confidences,
            )
        )
        return detections

    def _collect_crops(self, result: Any) -> tuple[list[Crop], Confidence]:
        crops: list[Crop] = []
        confidences: Confidence = {}
        for box in result.boxes:
            class_id = int(box.cls.item())
            class_name, threshold = self.class_confidences[class_id]
            confidence = float(box.conf.item())
            if confidence < threshold:
                continue

            confidences[class_name] = max(confidences.get(class_name, 0), confidence)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crops.append(
                Crop(x1, y1, x2, y2, label=class_name, confidence=confidence)
            )
        return crops, confidences


class YoloRunner:
    model: YOLO
    class_confidences: dict[int, tuple[str, float]]
    mapper: YoloResultMapper

    def __init__(
        self,
        config: YoloConfig,
        onnx_config: OnnxConfig,
        source_count: int,
    ):
        self.config = config
        _patch_windows_path_checkpoints()
        self.model = YOLO(
            self._model_path(config, onnx_config, source_count),
            task=config.task,
        )
        self._setup_predictor()
        self.class_confidences = self._resolve_class_confidences(config.confidence)
        self.mapper = YoloResultMapper(self.class_confidences)

    def _model_path(
        self, config: YoloConfig, onnx_config: OnnxConfig, source_count: int
    ) -> str:
        if config.model.endswith(".onnx") or TYPE == "cuda":
            return config.model

        return str(
            YOLO(config.model, task=config.task).export(
                format="engine" if TYPE == "tensorrt" else "onnx",
                batch=max(1, source_count),
                dynamic=True,
                half=should_half(),
                imgsz=config.imgsz,
                simplify=True,
                opset=onnx_config.opset,
            )
        )

    def _setup_predictor(self) -> None:
        if self.model.predictor is not None:
            return
        self.model.predictor = self.model._smart_load("predictor")(
            overrides=self.model.overrides,
            _callbacks=self.model.callbacks,
        )
        self.model.predictor.setup_model(model=self.model.model, verbose=False)

    def predict(self, frames: list[ndarray]) -> list[Any]:
        then = datetime.now()
        results = self.model.predict(
            source=frames,
            conf=min_confidence(self.config.confidence),
            stream=False,
            classes=list(self.class_confidences.keys()) or None,
            imgsz=self.config.imgsz,
            rect=should_rect(),
            batch=len(frames),
        )
        now = datetime.now()
        logger.info(
            "Detection time: %dms for %d frame(s). Avg: %dms",
            (now - then).total_seconds() * 1000,
            len(frames),
            (now - then).total_seconds() * 1000 / len(frames),
        )
        return list(results)

    def detections_from_result(
        self, result: Any, frames: list[tuple[datetime, ndarray]]
    ) -> list[Detection] | None:
        return self.mapper.detections_from_result(result, frames)

    def _resolve_class_confidences(
        self, confidence: float | dict[str, float]
    ) -> dict[int, tuple[str, float]]:
        yolo_names = self.model.names
        id_to_name = (
            {
                int(class_id): str(class_name)
                for class_id, class_name in yolo_names.items()
            }
            if isinstance(yolo_names, dict)
            else {
                class_id: str(class_name)
                for class_id, class_name in enumerate(yolo_names)
            }
        )

        if not isinstance(confidence, dict):
            threshold = float(confidence)
            return {
                class_id: (class_name, threshold)
                for class_id, class_name in id_to_name.items()
            }

        name_to_id = {
            class_name: class_id for class_id, class_name in id_to_name.items()
        }
        class_confidences: dict[int, tuple[str, float]] = {}
        for raw_class_name, threshold in confidence.items():
            class_name = raw_class_name.strip()
            class_id = name_to_id.get(class_name)
            if class_id is None:
                available_names = ", ".join(
                    id_to_name[class_id] for class_id in sorted(id_to_name)
                )
                raise ValueError(
                    f"Unknown YOLO class name '{raw_class_name}' in yolo.confidence. "
                    f"Available class names: {available_names}"
                )
            class_confidences[class_id] = (id_to_name[class_id], float(threshold))

        return class_confidences
