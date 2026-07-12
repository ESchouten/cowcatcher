from dataclasses import dataclass

from aidetector.domain.detections import ConfidenceThreshold, confidence_matches
from aidetector.domain.events import DetectionEvent

Cooldown = float | dict[str, float]


@dataclass(frozen=True, slots=True)
class EventPolicy:
    frames_min: int
    timeout: float
    time_max: float
    trailing_time: float


@dataclass(frozen=True, slots=True)
class CooldownPolicy:
    confidence: ConfidenceThreshold
    cooldown: Cooldown

    def for_class(self, class_name: str) -> float:
        if isinstance(self.cooldown, dict):
            return self.cooldown.get(class_name, 0)
        return self.cooldown


@dataclass(frozen=True, slots=True)
class ExportPolicy:
    confidence: ConfidenceThreshold
    export_rejected: bool

    def accepts(self, event: DetectionEvent, validated: bool | None) -> bool:
        return confidence_matches(event.best.confidences, self.confidence) and (
            validated is not False or self.export_rejected
        )
