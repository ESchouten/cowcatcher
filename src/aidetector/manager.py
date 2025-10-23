from typing import Self

from aidetector.config import Config
from aidetector.detector import Detector


class Manager:
    def __init__(self, detectors: list[Detector]):
        self.detectors = detectors

    @classmethod
    def fromConfig(cls, config: Config) -> Self:
        return cls(
            [Detector.fromConfig(config, detector) for detector in config.detectors]
        )

    def start(self):
        for detector in self.detectors:
            detector.start()
