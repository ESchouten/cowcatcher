from typing import Protocol

from numpy import ndarray

from aidetector.utils.config import IdentityProviderConfig, IdentityResult


class IdentityProvider(Protocol):
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
