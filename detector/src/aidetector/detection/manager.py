from typing_extensions import Self

from aidetector.utils.config import Config
from aidetector.detection.detector import Detector
from aidetector.identity.service import IdentityService
from aidetector.services.healthcheck import Healthcheck


class Manager:
    detectors: list[Detector]
    health: Healthcheck | None
    identity_service: IdentityService | None

    def __init__(
        self,
        detectors: list[Detector],
        health: Healthcheck | None,
        identity_service: IdentityService | None = None,
    ):
        self.detectors = detectors
        self.health = health
        self.identity_service = identity_service

    @classmethod
    def from_config(cls, config: Config) -> Self:
        identity_service = IdentityService.from_config(config)
        detectors = [
            d
            for ds in [
                Detector.from_config(config, detector, identity_service)
                for detector in config.detectors
            ]
            for d in ds
        ]
        health = Healthcheck(config.health) if config.health else None
        return cls(detectors, health, identity_service)

    def start(self):
        threads = [detector.start() for detector in self.detectors]
        if self.health:
            threads.append(self.health.start())
        return threads

    def stop(self):
        if self.health:
            self.health.stop()
        if self.identity_service:
            self.identity_service.close()
