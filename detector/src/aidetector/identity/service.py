from typing import Protocol

from aidetector.utils.config import Config, Detection, DetectorIdentityConfig, IdentityResult

IdentityLookupResult = IdentityResult | list[IdentityResult] | None


class IdentityProvider(Protocol):
    def identify(
        self,
        detection: Detection | list[Detection],
        source: str,
        multiple: bool = False,
    ) -> IdentityLookupResult:
        pass

    def close(self) -> None:
        pass


class IdentityService:
    def __init__(self, providers: dict[str, IdentityProvider]):
        self.providers = providers

    @classmethod
    def from_config(cls, config: Config) -> "IdentityService | None":
        provider_configs = [
            detector.identity for detector in config.detectors if detector.identity is not None
        ]
        if not provider_configs:
            return None

        from aidetector.identity.wildlife_tools import WildlifeToolsIdentityProvider

        providers: dict[str, IdentityProvider] = {}
        provider_signatures: dict[str, tuple[tuple[str, object], ...]] = {}
        for provider_config in provider_configs:
            signature = _provider_signature(provider_config)
            if provider_config.id in providers:
                if provider_signatures[provider_config.id] != signature:
                    raise ValueError(
                        "Duplicate identity provider id with conflicting "
                        f"configuration: {provider_config.id}"
                    )
                continue
            if provider_config.type == "wildlife_tools":
                providers[provider_config.id] = WildlifeToolsIdentityProvider(
                    provider_config
                )
                provider_signatures[provider_config.id] = signature
            else:
                raise ValueError(f"Unknown identity provider type: {provider_config.type}")
        return cls(providers)

    def identify(
        self,
        provider: str,
        detection: Detection | list[Detection],
        source: str,
        multiple: bool = False,
    ) -> IdentityLookupResult:
        identity_provider = self.providers.get(provider)
        if identity_provider is None:
            raise ValueError(f"Unknown identity provider id: {provider}")
        return identity_provider.identify(detection, source, multiple)

    def close(self) -> None:
        for provider in self.providers.values():
            provider.close()


def _provider_signature(config: DetectorIdentityConfig) -> tuple[tuple[str, object], ...]:
    return tuple(
        (key, getattr(config, key))
        for key in (
            "id",
            "type",
            "database",
            "model",
            "segment_model",
            "segment_labels",
            "segment_confidence",
            "segment_background",
            "debug_directory",
            "match_threshold",
            "candidate_threshold",
            "create_after",
            "crop_padding",
        )
    )
