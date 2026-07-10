from dataclasses import asdict

from aidetector.detection.models import Detection, max_confidence
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass


@dataclass(config=ConfigDict(extra="forbid"))
class CropMetadata:
    x1: int
    y1: int
    x2: int
    y2: int


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

    @classmethod
    def from_event(
        cls,
        timestamp: str,
        best: Detection,
        detections: list[Detection],
        validated: bool | None,
    ) -> "DetectionMetadata":
        region = best.images.crop_region
        return cls(
            timestamp=timestamp,
            validated=validated,
            confidence=max_confidence(best.confidence),
            confidences=best.confidence,
            detections=len(detections),
            start=detections[0].date.isoformat(),
            end=detections[-1].date.isoformat(),
            duration=(detections[-1].date - detections[0].date).total_seconds(),
            crop=CropMetadata(region.x1, region.y1, region.x2, region.y2)
            if region
            else None,
        )

    def as_dict(self) -> dict:
        return asdict(self)
