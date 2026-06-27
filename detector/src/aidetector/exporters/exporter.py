import logging
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from aidetector.utils.config import (
    Detection,
    ExporterConfig,
    confidence_matches,
)

T = TypeVar("T", bound=ExporterConfig)


class Exporter(ABC, Generic[T]):
    logger = logging.getLogger(__name__)
    config: T

    def __init__(self, config: T):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing")
        self.config = config

    def export(
        self,
        best_detection: Detection,
        detections: list[Detection],
        validated: bool | None,
    ):
        if not confidence_matches(best_detection.confidence, self.config.confidence or 0):
            self.logger.info("Confidence does not match")
            return
        if validated is False and not self.config.export_rejected:
            self.logger.info("Best detection is rejected and export_rejected is False")
            return
        self.filtered_export(best_detection, detections, validated)

    @abstractmethod
    def filtered_export(
        self,
        best_detection: Detection,
        detections: list[Detection],
        validated: bool | None,
    ):
        pass
