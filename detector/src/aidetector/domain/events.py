from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4

from aidetector.domain.detections import Observation, max_confidence


@dataclass(frozen=True, slots=True)
class LiveObservation:
    source: str
    observation: Observation


@dataclass(frozen=True, slots=True)
class DetectionEvent:
    source: str
    observations: tuple[Observation, ...]
    best: Observation
    event_id: str = field(default_factory=lambda: uuid4().hex)

    @property
    def started_at(self) -> datetime:
        return self.observations[0].frame.captured_at

    @property
    def ended_at(self) -> datetime:
        return self.observations[-1].frame.captured_at

    @property
    def duration(self) -> float:
        return (self.ended_at - self.started_at).total_seconds()

    @property
    def confidence(self) -> float:
        return max_confidence(self.best.confidences)
