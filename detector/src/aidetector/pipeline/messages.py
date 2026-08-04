from dataclasses import dataclass
from typing import Literal

from aidetector.domain.events import DetectionEvent
from aidetector.pipeline.ports import ArtifactProvider

ValidationStatus = Literal["approved", "rejected", "unvalidated"]


@dataclass(frozen=True, slots=True)
class CompletedEvent:
    event: DetectionEvent
    validated: bool | None
    artifacts: ArtifactProvider

    @property
    def status(self) -> ValidationStatus:
        if self.validated is True:
            return "approved"
        if self.validated is False:
            return "rejected"
        return "unvalidated"
