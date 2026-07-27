from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

IdentityStatus = Literal[
    "matched",
    "unknown",
    "ambiguous",
    "insufficient_evidence",
    "switch_risk",
    "error",
]


@dataclass(frozen=True, slots=True)
class IdentityResult:
    status: IdentityStatus
    visual_identity_id: str | None = None
    official_id: str | None = None
    similarity: float | None = None
    margin: float | None = None
    gallery_version: int | None = None

    def __post_init__(self) -> None:
        if (
            self.visual_identity_id is not None
            and not self.visual_identity_id.startswith("vid_")
        ):
            raise ValueError("Visual identity IDs must use the vid_ namespace")
        if self.official_id is not None and not self.official_id.strip():
            raise ValueError("Official identity IDs must not be empty")
        for name, value in (
            ("similarity", self.similarity),
            ("margin", self.margin),
        ):
            if value is not None and not math.isfinite(value):
                raise ValueError(f"Identity {name} must be finite")
        if self.gallery_version is not None and self.gallery_version < 1:
            raise ValueError("Gallery version must be positive")
        if self.status == "matched":
            if (
                self.visual_identity_id is None
                or self.official_id is None
                or self.similarity is None
                or self.margin is None
                or self.gallery_version is None
            ):
                raise ValueError("Matched identity results must be complete")
        elif self.official_id is not None:
            raise ValueError("Only matched results may expose an official identity")
