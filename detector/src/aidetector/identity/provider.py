from typing import Protocol

from numpy import ndarray

from aidetector.utils.config import Config, IdentityProviderConfig, IdentityResult


class IdentityProvider(Protocol):
    def match(self, image: ndarray) -> IdentityResult | None:
        pass

    def identify(self, image: ndarray) -> IdentityResult | None:
        pass

    def update_identity(
        self,
        identity: str,
        image: ndarray,
    ) -> IdentityResult | None:
        pass

    def close(self) -> None:
        pass


def create_identity_provider(config: IdentityProviderConfig) -> IdentityProvider:
    if config.type == "wildlife_tools":
        from aidetector.identity.wildlife_tools import WildlifeToolsIdentityProvider

        return WildlifeToolsIdentityProvider(config)
    raise ValueError(f"Unknown identity provider type: {config.type}")


def create_identity_providers(config: Config) -> dict[str, IdentityProvider]:
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
