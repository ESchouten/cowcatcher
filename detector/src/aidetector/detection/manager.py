from typing_extensions import Self

from aidetector.detection.detector import Detector
from aidetector.identity.provider import (
    IdentityProvider,
    create_identity_providers,
)
from aidetector.services.healthcheck import Healthcheck
from aidetector.utils.config import Config


class Manager:
    detectors: list[Detector]
    health: Healthcheck | None
    identity_providers: dict[str, IdentityProvider]

    def __init__(
        self,
        detectors: list[Detector],
        health: Healthcheck | None,
        identity_providers: dict[str, IdentityProvider] | None = None,
    ):
        self.detectors = detectors
        self.health = health
        self.identity_providers = identity_providers or {}

    @classmethod
    def from_config(cls, config: Config) -> Self:
        identity_providers = create_identity_providers(config)
        detectors = [
            d
            for ds in [
                Detector.from_config(config, detector, identity_providers)
                for detector in config.detectors
            ]
            for d in ds
        ]
        health = Healthcheck(config.health) if config.health else None
        return cls(detectors, health, identity_providers)

    def start(self):
        threads = [detector.start() for detector in self.detectors]
        if self.health:
            threads.append(self.health.start())
        return threads

    def stop(self):
        if self.health:
            self.health.stop()
        for provider in self.identity_providers.values():
            provider.close()
