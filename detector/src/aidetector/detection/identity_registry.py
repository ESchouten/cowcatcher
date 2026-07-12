from collections.abc import Callable
from dataclasses import dataclass
from threading import Lock
from time import monotonic

from aidetector.utils.config import Crop, Detection, IdentityResult


@dataclass(frozen=True)
class _Observation:
    x1: float
    y1: float
    x2: float
    y2: float
    identities: tuple[IdentityResult, ...]
    updated_at: float


class IdentityRegistry:
    def __init__(
        self,
        *,
        ttl: float = 5,
        clock: Callable[[], float] = monotonic,
    ):
        if ttl <= 0:
            raise ValueError("Identity registry TTL must be positive")
        self.ttl = ttl
        self.clock = clock
        self._observations: dict[tuple[str, str, int], _Observation] = {}
        self._lock = Lock()

    def publish(self, source: str, producer: str, detection: Detection) -> None:
        height, width = detection.images.jpg.shape[:2]
        if width == 0 or height == 0:
            return
        now = self.clock()
        observations = {
            (source, producer, crop.track_id): _Observation(
                crop.x1 / width,
                crop.y1 / height,
                crop.x2 / width,
                crop.y2 / height,
                tuple(crop.identities),
                now,
            )
            for crop in detection.images.crops
            if crop.track_id is not None and crop.identities
        }
        with self._lock:
            self._prune(now)
            self._observations.update(observations)

    def enrich(self, source: str, detection: Detection) -> None:
        height, width = detection.images.jpg.shape[:2]
        if width == 0 or height == 0:
            return
        now = self.clock()
        with self._lock:
            self._prune(now)
            observations = [
                observation
                for (
                    observation_source,
                    _,
                    _,
                ), observation in self._observations.items()
                if observation_source == source
            ]

        for crop in detection.images.crops:
            identities = list(crop.identities)
            seen = {identity.identity for identity in identities}
            target = _normalize_crop(crop, width, height)
            for observation in observations:
                if not _contains_center(target, observation):
                    continue
                for identity in observation.identities:
                    if identity.identity in seen:
                        continue
                    identities.append(identity)
                    seen.add(identity.identity)
            crop.identities = identities

    def _prune(self, now: float) -> None:
        expired = [
            key
            for key, observation in self._observations.items()
            if now - observation.updated_at > self.ttl
        ]
        for key in expired:
            del self._observations[key]


def _normalize_crop(crop: Crop, width: int, height: int) -> tuple[float, ...]:
    return crop.x1 / width, crop.y1 / height, crop.x2 / width, crop.y2 / height


def _contains_center(
    target: tuple[float, ...],
    observation: _Observation,
) -> bool:
    center_x = (observation.x1 + observation.x2) / 2
    center_y = (observation.y1 + observation.y2) / 2
    return target[0] <= center_x <= target[2] and target[1] <= center_y <= target[3]
