from dataclasses import asdict, field

from aidetector.domain.detections import DetectedObject
from aidetector.domain.events import DetectionEvent
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass


@dataclass(config=ConfigDict(extra="forbid"))
class CropMetadata:
    x1: int
    y1: int
    x2: int
    y2: int
    label: str | None = None
    confidence: float | None = None

    @classmethod
    def from_object(cls, item: DetectedObject) -> "CropMetadata":
        return cls(
            x1=item.x1,
            y1=item.y1,
            x2=item.x2,
            y2=item.y2,
            label=item.label,
            confidence=item.confidence,
        )

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(config=ConfigDict(extra="forbid"))
class DetectionMetadata:
    event_id: str
    source: str
    timestamp: str
    validated: bool | None
    confidence: float
    confidences: dict[str, float]
    observations: int
    start: str
    end: str
    duration: float
    crop: CropMetadata | None = None
    crops: list[CropMetadata] = field(default_factory=list)

    @classmethod
    def from_event(
        cls,
        event: DetectionEvent,
        validated: bool | None,
    ) -> "DetectionMetadata":
        region = event.best.crop_region
        return cls(
            event_id=event.event_id,
            source=event.source,
            timestamp=event.best.frame.captured_at.isoformat(),
            validated=validated,
            confidence=event.confidence,
            confidences=dict(event.best.confidences),
            observations=len(event.observations),
            start=event.started_at.isoformat(),
            end=event.ended_at.isoformat(),
            duration=event.duration,
            crop=CropMetadata.from_object(region) if region else None,
            crops=[CropMetadata.from_object(item) for item in event.best.objects],
        )

    def as_dict(self) -> dict:
        return asdict(self)
