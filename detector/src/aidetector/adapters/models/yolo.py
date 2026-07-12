import logging
from time import perf_counter
from typing import Any

import numpy as np
from aidetector.domain.detections import (
    DetectedObject,
    Observation,
    min_confidence,
)
from aidetector.domain.frames import Frame, FrameBatch
from aidetector.pipeline.ports import ModelBatchResult
from aidetector.utils.config import (
    OnnxConfig,
    YoloConfig,
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


class YoloResultMapper:
    def __init__(self, class_confidences: dict[int, tuple[str, float]]):
        self.class_confidences = class_confidences

    def observations_from_result(
        self, result: Any, frames: list[Frame]
    ) -> list[Observation] | None:
        objects, confidences = self._collect_objects(result)
        if not objects:
            return None

        observations = [Observation(frame) for frame in frames[:-1]]
        observations.append(Observation(frames[-1], tuple(objects), confidences))
        return observations

    def _collect_objects(
        self, result: Any
    ) -> tuple[list[DetectedObject], dict[str, float]]:
        objects: list[DetectedObject] = []
        confidences: dict[str, float] = {}
        for box in result.boxes:
            class_id = int(box.cls.item())
            class_settings = self.class_confidences.get(class_id)
            if class_settings is None:
                continue
            class_name, threshold = class_settings
            confidence = float(box.conf.item())
            if confidence < threshold:
                continue

            confidences[class_name] = max(confidences.get(class_name, 0), confidence)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            raw_track_id = getattr(box, "id", None)
            track_id = int(raw_track_id.item()) if raw_track_id is not None else None
            objects.append(
                DetectedObject(
                    x1,
                    y1,
                    x2,
                    y2,
                    label=class_name,
                    confidence=confidence,
                    track_id=track_id,
                )
            )
        return objects, confidences


def build_yolo_model(
    config: YoloConfig,
    onnx_config: OnnxConfig,
    source_count: int,
) -> YOLO:
    model_path = config.model
    if not config.model.endswith(".onnx") and TYPE != "cuda":
        model_path = str(
            YOLO(config.model, task=config.task).export(
                format="engine" if TYPE == "tensorrt" else "onnx",
                batch=source_count,
                dynamic=True,
                half=should_half(),
                imgsz=config.imgsz,
                simplify=True,
                opset=onnx_config.opset,
            )
        )

    model = YOLO(model_path, task=config.task)
    _setup_ultralytics_predictor(model)
    return model


def _setup_ultralytics_predictor(model: YOLO) -> None:
    if model.predictor is not None:
        return
    model.predictor = model._smart_load("predictor")(
        overrides=model.overrides,
        _callbacks=model.callbacks,
    )
    model.predictor.setup_model(model=model.model, verbose=False)


class YoloRunner:
    model: YOLO
    sources: list[str]
    tracking_last_frames: dict[str, Frame]
    class_confidences: dict[int, tuple[str, float]]
    mapper: YoloResultMapper

    def __init__(
        self,
        config: YoloConfig,
        sources: list[str],
        model: YOLO,
    ):
        self.config = config
        self.sources = list(sources)
        self.model = model
        self.tracking_last_frames = {}
        self.class_confidences = self._resolve_class_confidences(config.confidence)
        self.mapper = YoloResultMapper(self.class_confidences)

    def detect(self, frames: list[ndarray]) -> list[Any]:
        then = perf_counter()
        results = self.model.predict(**self._predict_kwargs(frames, len(frames)))
        self._log_predict_time(then, len(frames), "Detection")
        return list(results)

    def track_sources(
        self,
        batch: FrameBatch,
    ) -> list[ModelBatchResult]:
        then = perf_counter()
        prepared = self._tracking_batch(batch)
        if prepared is None:
            return []

        sources, frames, active_frames = prepared
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

        tracked: list[ModelBatchResult] = []
        for source, result in zip(sources, results, strict=True):
            frames_for_source = active_frames.get(source)
            if frames_for_source is None:
                continue
            tracked.append(ModelBatchResult(source, result, frames_for_source))
        return tracked

    def _tracking_batch(
        self,
        batch: FrameBatch,
    ) -> (
        tuple[
            list[str],
            list[ndarray],
            FrameBatch,
        ]
        | None
    ):
        active_frames = {source: frames for source, frames in batch.items() if frames}
        if not active_frames:
            return None

        placeholder = np.zeros_like(
            next(iter(active_frames.values()))[-1].require_image()
        )
        images: list[ndarray] = []
        for source in self.sources:
            frames = active_frames.get(source)
            if frames is not None:
                latest = frames[-1]
                self.tracking_last_frames[source] = latest
                images.append(latest.require_image())
                continue

            previous = self.tracking_last_frames.get(source)
            images.append(
                previous.require_image() if previous is not None else placeholder
            )

        return self.sources, images, active_frames

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

    def _log_predict_time(self, then: float, frame_count: int, operation: str) -> None:
        elapsed = perf_counter() - then
        logger.info(
            "%s time: %dms for %d frame(s). Avg: %dms",
            operation,
            elapsed * 1000,
            frame_count,
            elapsed * 1000 / frame_count,
        )

    def observations_from_result(
        self, result: Any, frames: list[Frame]
    ) -> list[Observation] | None:
        return self.mapper.observations_from_result(result, frames)

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
