from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

IdentityStatus = Literal[
    "matched",
    "unknown",
    "ambiguous",
    "insufficient_evidence",
    "switch_risk",
]


@dataclass(frozen=True, slots=True)
class IdentityResult:
    status: IdentityStatus
    visual_identity_id: str | None = None
    official_id: str | None = None
    similarity: float | None = None
    margin: float | None = None
    gallery_version: int | None = None
