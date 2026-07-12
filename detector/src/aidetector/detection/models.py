from dataclasses import dataclass, field
from datetime import datetime

from numpy import ndarray

Confidence = dict[str, float]
ConfidenceThreshold = float | Confidence
Frame = tuple[datetime, ndarray]
FrameBatch = dict[str, list[Frame]]


@dataclass
class Crop:
    x1: int
    y1: int
    x2: int
    y2: int
    label: str | None = None
    confidence: float | None = None
    track_id: int | None = None


@dataclass
class ImageSet:
    jpg: ndarray
    crops: list[Crop] = field(default_factory=list)

    @property
    def crop_region(self) -> Crop | None:
        if not self.crops:
            return None
        if len(self.crops) == 1:
            return self.crops[0]
        return Crop(
            min(crop.x1 for crop in self.crops),
            min(crop.y1 for crop in self.crops),
            max(crop.x2 for crop in self.crops),
            max(crop.y2 for crop in self.crops),
        )


@dataclass
class Detection:
    date: datetime
    images: ImageSet
    confidence: Confidence

    @classmethod
    def from_frame(
        cls,
        frame: Frame,
        *,
        crops: list[Crop] | None = None,
        confidence: Confidence | None = None,
    ) -> "Detection":
        date, image = frame
        return cls(date, ImageSet(image, list(crops or [])), confidence or {})


def min_confidence(confidence: ConfidenceThreshold | None) -> float:
    if not confidence:
        return 0
    if isinstance(confidence, dict):
        return min(map(float, confidence.values()))
    return float(confidence)


def max_confidence(confidence: ConfidenceThreshold | None) -> float:
    if not confidence:
        return 0
    if isinstance(confidence, dict):
        return max(map(float, confidence.values()))
    return float(confidence)


def confidence_matches(
    value: Confidence,
    threshold: ConfidenceThreshold,
) -> bool:
    if isinstance(threshold, dict):
        return any(
            class_name in value and value[class_name] >= class_threshold
            for class_name, class_threshold in threshold.items()
        )
    return max_confidence(value) >= threshold


def matching_confidences(
    value: Confidence,
    threshold: ConfidenceThreshold,
) -> list[str]:
    if not isinstance(threshold, dict):
        return [
            class_name
            for class_name, confidence in value.items()
            if confidence >= threshold
        ]
    return [
        class_name
        for class_name, confidence in value.items()
        if class_name in threshold and confidence >= threshold[class_name]
    ]
