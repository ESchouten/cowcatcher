from typing_extensions import Self

from aidetector.detection.detector import Detector
from aidetector.detection.identity_registry import IdentityRegistry
from aidetector.services.healthcheck import Healthcheck
from aidetector.utils.config import Config


class Manager:
    detectors: list[Detector]
    health: Healthcheck | None

    def __init__(
        self,
        detectors: list[Detector],
        health: Healthcheck | None,
        identity_registry: IdentityRegistry | None = None,
    ):
        self.detectors = detectors
        self.health = health
        self.identity_registry = identity_registry or IdentityRegistry()
        for detector in detectors:
            detector.identity_registry = self.identity_registry

    @classmethod
    def from_config(cls, config: Config) -> Self:
        identity_registry = IdentityRegistry()
        detectors = [
            d
            for ds in [
                Detector.from_config(
                    config,
                    detector,
                    detector_index,
                    identity_registry,
                )
                for detector_index, detector in enumerate(config.detectors)
            ]
            for d in ds
        ]
        health = Healthcheck(config.health) if config.health else None
        return cls(detectors, health, identity_registry)

    def start(self):
        threads = [detector.start() for detector in self.detectors]
        if self.health:
            threads.append(self.health.start())
        return threads

    def stop(self):
        if self.health:
            self.health.stop()
