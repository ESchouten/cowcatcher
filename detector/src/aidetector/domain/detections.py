from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from aidetector.domain.frames import Frame

Number = int | float
Confidence = Mapping[str, Number]
ConfidenceThreshold = Number | dict[str, Number]
ConfidenceValue = Number | Confidence


@dataclass(frozen=True, slots=True)
class DetectedObject:
    x1: int
    y1: int
    x2: int
    y2: int
    label: str | None = None
    confidence: float | None = None
    track_id: int | None = None


@dataclass(frozen=True, slots=True)
class Observation:
    frame: Frame
    objects: tuple[DetectedObject, ...] = ()
    confidences: Confidence = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "confidences",
            MappingProxyType(dict(self.confidences)),
        )

    @property
    def crop_region(self) -> DetectedObject | None:
        return bounding_region(self.objects)


def bounding_region(objects: Iterable[DetectedObject]) -> DetectedObject | None:
    items = tuple(objects)
    if not items:
        return None
    if len(items) == 1:
        return items[0]
    return DetectedObject(
        min(item.x1 for item in items),
        min(item.y1 for item in items),
        max(item.x2 for item in items),
        max(item.y2 for item in items),
    )


def min_confidence(confidence: ConfidenceValue | None) -> float:
    if not confidence:
        return 0
    if isinstance(confidence, Mapping):
        return float(min(confidence.values()))
    return float(confidence)


def max_confidence(confidence: ConfidenceValue | None) -> float:
    if not confidence:
        return 0
    if isinstance(confidence, Mapping):
        return float(max(confidence.values()))
    return float(confidence)


def confidence_matches(value: Confidence, threshold: ConfidenceThreshold) -> bool:
    return bool(matching_confidences(value, threshold))


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
