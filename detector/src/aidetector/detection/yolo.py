import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import cv2
import numpy as np
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


@dataclass
class YoloObject:
    crop: Crop
    mask: np.ndarray | None = None
    track_id: int | None = None

    @property
    def area(self) -> int:
        if self.mask is not None:
            return int(self.mask.sum())
        return max(0, self.crop.x2 - self.crop.x1) * max(0, self.crop.y2 - self.crop.y1)


def objects_from_result(
    result: Any,
    image_shape: tuple[int, int],
    class_confidences: dict[int, tuple[str, float]] | None = None,
    labels: list[str] | None = None,
    min_confidence: float = 0,
) -> list[YoloObject]:
    if result.boxes is None:
        return []

    names = getattr(result, "names", {}) or {}
    accepted_labels = set(labels) if labels is not None else None
    objects: list[YoloObject] = []

    for index, box in enumerate(result.boxes):
        class_id = int(box.cls.item())
        class_name = _class_name(names, class_id)
        confidence = float(box.conf.item())
        threshold = min_confidence

        if class_confidences is not None:
            if class_id not in class_confidences:
                continue
            class_name, threshold = class_confidences[class_id]

        if accepted_labels is not None and class_name not in accepted_labels:
            continue
        if confidence < threshold:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        mask = _mask_from_result(result, index, image_shape)
        objects.append(
            YoloObject(
                crop=Crop(
                    x1,
                    y1,
                    x2,
                    y2,
                    label=class_name,
                    confidence=confidence,
                ),
                mask=mask,
                track_id=_track_id_from_result(result, index),
            )
        )

    return objects


def confidences_from_objects(objects: list[YoloObject]) -> Confidence:
    confidences: Confidence = {}
    for obj in objects:
        crop = obj.crop
        if crop.label is None or crop.confidence is None:
            continue
        confidences[crop.label] = max(
            confidences.get(crop.label, 0),
            crop.confidence,
        )
    return confidences


def apply_mask(
    image: np.ndarray,
    mask: np.ndarray,
    background: str,
) -> np.ndarray:
    background_value = 0 if background == "black" else 127
    masked = np.full_like(image, background_value)
    masked[mask] = image[mask]
    return masked


def _mask_from_result(
    result: Any,
    index: int,
    image_shape: tuple[int, int],
) -> np.ndarray | None:
    masks = getattr(result, "masks", None)
    if masks is None or index >= len(masks.data):
        return None

    mask = masks.data[index].detach().cpu().numpy().astype(bool)
    if mask.shape == image_shape:
        return mask

    return cv2.resize(
        mask.astype(np.uint8),
        (image_shape[1], image_shape[0]),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)


def _track_id_from_result(result: Any, index: int) -> int | None:
    track_ids = getattr(result.boxes, "id", None)
    if track_ids is None:
        return None

    try:
        return int(track_ids[index].item())
    except AttributeError:
        return int(track_ids[index])


def _class_name(names: Any, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


class YoloResultMapper:
    def __init__(self, class_confidences: dict[int, tuple[str, float]]):
        self.class_confidences = class_confidences

    def detections_from_result(
        self, result: Any, frames: list[tuple[datetime, ndarray]]
    ) -> list[Detection] | None:
        image_shape = getattr(result, "orig_shape", frames[-1][1].shape[:2])
        crops, confidences = self._collect_crops(result, image_shape)
        if not crops:
            return None

        detections = [
            Detection(frame_date, ImageSet(frame), {})
            for frame_date, frame in frames[:-1]
        ]
        detections.append(
            Detection(
                frames[-1][0],
                ImageSet(frames[-1][1], _clone_crops(crops)),
                confidences,
            )
        )
        return detections

    def _collect_crops(
        self,
        result: Any,
        image_shape: tuple[int, int],
    ) -> tuple[list[Crop], Confidence]:
        objects = sorted(
            objects_from_result(
                result,
                image_shape,
                class_confidences=self.class_confidences,
            ),
            key=lambda obj: obj.crop.confidence or 0,
            reverse=True,
        )
        return [obj.crop for obj in objects], confidences_from_objects(objects)


def _clone_crops(crops: list[Crop]) -> list[Crop]:
    return [crop.clone() for crop in crops]


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
