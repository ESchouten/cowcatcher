import logging
from abc import ABC, abstractmethod
from typing import Generic, Protocol, TypeVar

from aidetector.detection.models import (
    Detection,
    confidence_matches,
)
from aidetector.detection.events import DetectionEvent
from aidetector.utils.config import ExporterConfig

T = TypeVar("T", bound=ExporterConfig)


class TrackPublisher(Protocol):
    def publish_tracks(self, source: str, detection: Detection) -> None: ...


class Exporter(ABC, Generic[T]):
    config: T

    def __init__(self, config: T):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing")
        self.config = config

    def export(
        self,
        event: DetectionEvent,
        validated: bool | None,
    ) -> bool:
        if not confidence_matches(event.best.confidence, self.config.confidence or 0):
            self.logger.info("Confidence does not match")
            return False
        if validated is False and not self.config.export_rejected:
            self.logger.info("Best detection is rejected and export_rejected is False")
            return False
        self._export(event, validated)
        return True

    @abstractmethod
    def _export(
        self,
        event: DetectionEvent,
        validated: bool | None,
    ) -> None: ...

    def close(self) -> None:
        pass
