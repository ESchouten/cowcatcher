from typing_extensions import Self

from aidetector.detection.detector import Detector
from aidetector.identity.provider import (
    IdentityProvider,
    create_identity_provider,
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
        identity_providers = _identity_providers_from_config(config)
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


def _identity_providers_from_config(config: Config) -> dict[str, IdentityProvider]:
    provider_configs = list(config.identity.providers if config.identity else [])
    detector_provider_ids = {
        detector.identity.provider
        for detector in config.detectors
        if detector.identity is not None
    }
    if not provider_configs:
        if detector_provider_ids:
            raise ValueError(
                "Detector identity references providers, but no identity providers "
                "are configured"
            )
        return {}

    provider_ids: set[str] = set()
    for provider_config in provider_configs:
        if provider_config.id in provider_ids:
            raise ValueError(f"Duplicate identity provider id: {provider_config.id}")
        provider_ids.add(provider_config.id)

    missing_provider_ids = detector_provider_ids - provider_ids
    if missing_provider_ids:
        raise ValueError(
            "Unknown identity provider id(s): " + ", ".join(sorted(missing_provider_ids))
        )

    return {
        provider_config.id: create_identity_provider(provider_config)
        for provider_config in provider_configs
    }
