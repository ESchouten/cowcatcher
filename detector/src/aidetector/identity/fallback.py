import logging
from dataclasses import dataclass
from threading import Lock

import numpy as np
from aidetector.detection.yolo import objects_from_result
from aidetector.utils.config import Crop, DetectedObject, IdentityFallbackConfig


@dataclass
class FallbackCandidates:
    detected: list[DetectedObject]
    matched: list[DetectedObject]
    selected: list[DetectedObject]


class FallbackCandidateExtractor:
    logger = logging.getLogger(__name__)

    def __init__(self, config: IdentityFallbackConfig):
        self.config = config
        self.model = None
        self.lock = Lock()

    def extract(
        self,
        image: np.ndarray,
        crop: Crop,
        multiple: bool,
    ) -> FallbackCandidates:
        detected = self._detect(image)
        labels = set(self.config.labels)
        matched = [
            obj
            for obj in detected
            if obj.mask is not None
            and _label_matches(obj.crop.label, labels)
            and (obj.crop.confidence or 0) >= self.config.confidence
        ]
        centered = sorted(
            [obj for obj in matched if _object_center_in_crop(obj, crop)],
            key=lambda obj: _center_distance_to_crop(obj, crop),
        )
        return FallbackCandidates(
            detected=detected,
            matched=matched,
            selected=centered if multiple else centered[:1],
        )

    def _detect(self, image: np.ndarray) -> list[DetectedObject]:
        from ultralytics import YOLO

        with self.lock:
            if self.model is None:
                self.model = YOLO(self.config.model, task="segment")
                self.logger.info(
                    "Loaded identity fallback segment model %s",
                    self.config.model,
                )

            results = self.model.predict(
                source=image,
                conf=self.config.confidence,
                imgsz=self.config.imgsz,
                stream=False,
                verbose=False,
            )
        if not results:
            return []
        return objects_from_result(results[0], image.shape[:2])


def _label_matches(label: str | None, labels: set[str]) -> bool:
    return label in labels


def _center_distance_to_crop(obj: DetectedObject, crop: Crop) -> float:
    target_x = (crop.x1 + crop.x2) / 2
    target_y = (crop.y1 + crop.y2) / 2
    crop_center_x = (obj.crop.x1 + obj.crop.x2) / 2
    crop_center_y = (obj.crop.y1 + obj.crop.y2) / 2
    return (crop_center_x - target_x) ** 2 + (crop_center_y - target_y) ** 2


def _object_center_in_crop(obj: DetectedObject, crop: Crop) -> bool:
    center_x = (obj.crop.x1 + obj.crop.x2) / 2
    center_y = (obj.crop.y1 + obj.crop.y2) / 2
    return crop.x1 <= center_x <= crop.x2 and crop.y1 <= center_y <= crop.y2
