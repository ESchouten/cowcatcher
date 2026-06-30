from typing import Protocol

from aidetector.utils.config import (
    Config,
    Detection,
    IdentityResult,
)

class IdentityProvider(Protocol):
    def identify(
        self,
        detection: Detection,
        source: str,
        multiple: bool = False,
    ) -> list[IdentityResult]:
        pass

    def close(self) -> None:
        pass


class IdentityService:
    def __init__(self, providers: dict[str, IdentityProvider]):
        self.providers = providers

    @classmethod
    def from_config(cls, config: Config) -> "IdentityService | None":
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
            return None

        from aidetector.identity.wildlife_tools import WildlifeToolsIdentityProvider

        providers: dict[str, IdentityProvider] = {}
        for provider_config in provider_configs:
            if provider_config.id in providers:
                raise ValueError(
                    f"Duplicate identity provider id: {provider_config.id}"
                )
            if provider_config.type == "wildlife_tools":
                providers[provider_config.id] = WildlifeToolsIdentityProvider(
                    provider_config
                )
            else:
                raise ValueError(f"Unknown identity provider type: {provider_config.type}")
        missing_provider_ids = detector_provider_ids - set(providers)
        if missing_provider_ids:
            raise ValueError(
                "Unknown identity provider id(s): "
                + ", ".join(sorted(missing_provider_ids))
            )
        return cls(providers)

    def identify(
        self,
        provider: str,
        detection: Detection,
        source: str,
        multiple: bool = False,
    ) -> list[IdentityResult]:
        identity_provider = self.providers.get(provider)
        if identity_provider is None:
            raise ValueError(f"Unknown identity provider id: {provider}")
        return identity_provider.identify(detection, source, multiple)

    def close(self) -> None:
        for provider in self.providers.values():
            provider.close()
