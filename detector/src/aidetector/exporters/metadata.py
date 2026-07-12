from dataclasses import asdict, field

from aidetector.detection.events import DetectionEvent
from aidetector.detection.models import Crop
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
    def from_crop(cls, crop: Crop) -> "CropMetadata":
        return cls(
            x1=crop.x1,
            y1=crop.y1,
            x2=crop.x2,
            y2=crop.y2,
            label=crop.label,
            confidence=crop.confidence,
        )

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(config=ConfigDict(extra="forbid"))
class DetectionMetadata:
    timestamp: str
    validated: bool | None
    confidence: float
    confidences: dict[str, float]
    detections: int
    start: str
    end: str
    duration: float
    crop: CropMetadata | None = None
    crops: list[CropMetadata] = field(default_factory=list)

    @classmethod
    def from_event(
        cls,
        timestamp: str,
        event: DetectionEvent,
        validated: bool | None,
    ) -> "DetectionMetadata":
        region = event.best.images.crop_region
        return cls(
            timestamp=timestamp,
            validated=validated,
            confidence=event.confidence,
            confidences=event.best.confidence,
            detections=len(event.detections),
            start=event.started_at.isoformat(),
            end=event.ended_at.isoformat(),
            duration=event.duration,
            crop=CropMetadata.from_crop(region) if region else None,
            crops=[CropMetadata.from_crop(crop) for crop in event.best.images.crops],
        )

    def as_dict(self) -> dict:
        return asdict(self)
