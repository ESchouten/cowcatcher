from typing import Protocol

from aidetector.utils.config import Detection, IdentityConfig, IdentityResult

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
    def from_config(cls, config: IdentityConfig | None) -> "IdentityService | None":
        if not config or not config.providers:
            return None

        from aidetector.identity.wildlife_tools import WildlifeToolsIdentityProvider

        providers: dict[str, IdentityProvider] = {}
        for provider_config in config.providers:
            if provider_config.id in providers:
                raise ValueError(f"Duplicate identity provider id: {provider_config.id}")
            if provider_config.type == "wildlife_tools":
                providers[provider_config.id] = WildlifeToolsIdentityProvider(
                    provider_config
                )
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
