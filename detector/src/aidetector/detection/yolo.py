import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

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
from ultralytics.data.loaders import LoadStreams, SourceTypes

logger = logging.getLogger(__name__)


class UltralyticsStreamBatch(LoadStreams):
    """Single in-memory batch that Ultralytics treats as a stream."""

    def __init__(self, paths: list[str], images: list[ndarray]):
        self.sources = paths
        self.images = images
        self.bs = len(images)
        self.mode = "stream"
        self.source_type = SourceTypes(stream=True)
        self.count = 0

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self) -> tuple[list[str], list[ndarray], list[str]]:
        if self.count > 0:
            raise StopIteration
        self.count += 1
        return self.sources, self.images, [""] * self.bs

    def __len__(self) -> int:
        return self.bs

    def close(self) -> None:
        pass


@dataclass
class TrackedSourceResult:
    source: str
    result: Any
    frames: list[tuple[datetime, ndarray]]


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
    model_path: str
    sources: list[str]
    tracking_last_frames: dict[str, tuple[datetime, ndarray]]
    class_confidences: dict[int, tuple[str, float]]
    mapper: YoloResultMapper

    def __init__(
        self,
        config: YoloConfig,
        onnx_config: OnnxConfig,
        sources: list[str],
    ):
        self.config = config
        self.sources = list(sources)
        self.model_path = self._model_path(config, onnx_config, len(self.sources))
        self.model = YOLO(self.model_path, task=config.task)
        self._setup_predictor()
        self.tracking_last_frames = {}
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

    def detect(self, frames: list[ndarray]) -> list[Any]:
        then = datetime.now()
        results = self.model.predict(**self._predict_kwargs(frames, len(frames)))
        self._log_predict_time(then, len(frames), "Detection")
        return list(results)

    def track_sources(
        self,
        batch: dict[str, list[tuple[datetime, ndarray]]],
    ) -> list[TrackedSourceResult]:
        then = datetime.now()
        prepared = self._tracking_batch(batch)
        if prepared is None:
            return []

        sources, frames, active_frames = prepared
        self._reset_tracking_if_batch_size_changed(len(frames))
        paths = [f"source-{index}" for index in range(len(sources))]
        results = list(
            self.model.track(
                **self._predict_kwargs(
                    UltralyticsStreamBatch(paths, frames),
                    len(frames),
                    stream=True,
                ),
                persist=True,
            )
        )
        self._log_predict_time(then, len(frames), "Tracking")

        tracked: list[TrackedSourceResult] = []
        for source, result in zip(sources, results):
            frames_for_source = active_frames.get(source)
            if frames_for_source is None:
                continue
            tracked.append(TrackedSourceResult(source, result, frames_for_source))
        return tracked

    def _tracking_batch(
        self,
        batch: dict[str, list[tuple[datetime, ndarray]]],
    ) -> (
        tuple[
            list[str],
            list[ndarray],
            dict[str, list[tuple[datetime, ndarray]]],
        ]
        | None
    ):
        active_frames = {source: frames for source, frames in batch.items() if frames}
        if not active_frames:
            return None

        placeholder = np.zeros_like(next(iter(active_frames.values()))[-1][1])
        ordered_sources = list(dict.fromkeys([*self.sources, *active_frames.keys()]))
        sources: list[str] = []
        images: list[ndarray] = []
        for source in ordered_sources:
            frames = active_frames.get(source)
            if frames is not None:
                latest = frames[-1]
                self.tracking_last_frames[source] = latest
                sources.append(source)
                images.append(latest[1])
                continue

            previous = self.tracking_last_frames.get(source)
            sources.append(source)
            images.append(previous[1] if previous is not None else placeholder)

        return sources, images, active_frames

    def _predict_kwargs(
        self,
        source,
        batch: int,
        stream: bool = False,
    ) -> dict[str, Any]:
        return {
            "source": source,
            "conf": min_confidence(self.config.confidence),
            "stream": stream,
            "classes": list(self.class_confidences.keys()) or None,
            "imgsz": self.config.imgsz,
            "rect": should_rect(),
            "batch": batch,
        }

    def _reset_tracking_if_batch_size_changed(self, batch_size: int) -> None:
        predictor = getattr(self.model, "predictor", None)
        trackers = getattr(predictor, "trackers", None)
        if trackers is None or len(trackers) == batch_size:
            return
        delattr(predictor, "trackers")
        if hasattr(predictor, "vid_path"):
            delattr(predictor, "vid_path")

    def _log_predict_time(
        self, then: datetime, frame_count: int, operation: str
    ) -> None:
        now = datetime.now()
        logger.info(
            "%s time: %dms for %d frame(s). Avg: %dms",
            operation,
            (now - then).total_seconds() * 1000,
            frame_count,
            (now - then).total_seconds() * 1000 / frame_count,
        )

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
