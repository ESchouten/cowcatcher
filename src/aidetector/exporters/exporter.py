import logging
from abc import ABC, abstractmethod
from typing import Self

from aidetector.config import Config, DetectorConfig


class Exporter(ABC):
    logger = logging.getLogger(__name__)

    def __init__(self, *args):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Initializing with args={args}")

    @classmethod
    @abstractmethod
    def fromConfig(cls: Self, config: Config, detector: DetectorConfig) -> Self | None:
        pass

    @abstractmethod
    def export(self, jpg: bytes):
        pass
