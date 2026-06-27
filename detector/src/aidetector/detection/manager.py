from typing_extensions import Self

from aidetector.utils.config import Config
from aidetector.detection.detector import Detector
from aidetector.services.healthcheck import Healthcheck


class Manager:
    detectors: list[Detector]
    health: Healthcheck | None

    def __init__(self, detectors: list[Detector], health: Healthcheck | None):
        self.detectors = detectors
        self.health = health

    @classmethod
    def from_config(cls, config: Config) -> Self:
        detectors = [
            d
            for ds in [
                Detector.from_config(config, detector) for detector in config.detectors
            ]
            for d in ds
        ]
        health = Healthcheck(config.health) if config.health else None
        return cls(detectors, health)

    def start(self):
        threads = [detector.start() for detector in self.detectors]
        if self.health:
            threads.append(self.health.start())
        return threads

    def stop(self):
        if self.health:
            self.health.stop()
