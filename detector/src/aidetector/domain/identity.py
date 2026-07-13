from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from numpy import ndarray

if TYPE_CHECKING:
    from aidetector.domain.detections import DetectedObject


@dataclass(frozen=True, slots=True)
class IdentityResult:
    identity: str
    similarity: float


@dataclass(frozen=True, slots=True)
class IdentityMatch:
    key: str
    label: str
    similarity: float
    margin: float

    @property
    def result(self) -> IdentityResult:
        return IdentityResult(self.label, self.similarity)


@dataclass(frozen=True, slots=True)
class IdentityCandidate:
    detection: DetectedObject
    image: ndarray


@dataclass(frozen=True, slots=True)
class TrackletSnapshot:
    source: str
    track_id: int
    first_frame: int
    last_frame: int
    observations: int
    embedding: ndarray
    preview: ndarray
    identity_key: str | None = None
