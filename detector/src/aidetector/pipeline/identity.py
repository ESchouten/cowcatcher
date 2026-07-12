from collections.abc import Callable
from dataclasses import dataclass, replace
from threading import Lock
from time import monotonic

from aidetector.domain.detections import DetectedObject, IdentityResult, Observation


@dataclass(frozen=True, slots=True)
class _IdentityObservation:
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
        self._observations: dict[tuple[str, str, int], _IdentityObservation] = {}
        self._lock = Lock()

    def publish(self, source: str, producer: str, observation: Observation) -> None:
        height, width = observation.frame.require_image().shape[:2]
        if not width or not height:
            return
        now = self.clock()
        identities = {
            (source, producer, item.track_id): _IdentityObservation(
                item.x1 / width,
                item.y1 / height,
                item.x2 / width,
                item.y2 / height,
                item.identities,
                now,
            )
            for item in observation.objects
            if item.track_id is not None and item.identities
        }
        with self._lock:
            self._prune(now)
            self._observations.update(identities)

    def enrich(self, source: str, observation: Observation) -> Observation:
        height, width = observation.frame.require_image().shape[:2]
        if not width or not height:
            return observation
        now = self.clock()
        with self._lock:
            self._prune(now)
            identities = [
                item
                for (item_source, _, _), item in self._observations.items()
                if item_source == source
            ]

        objects = tuple(
            _enrich_object(item, width, height, identities)
            for item in observation.objects
        )
        return replace(observation, objects=objects)

    def _prune(self, now: float) -> None:
        expired = [
            key
            for key, item in self._observations.items()
            if now - item.updated_at > self.ttl
        ]
        for key in expired:
            del self._observations[key]


def _enrich_object(
    target: DetectedObject,
    width: int,
    height: int,
    observations: list[_IdentityObservation],
) -> DetectedObject:
    identities = {item.identity: item for item in target.identities}
    normalized = (
        target.x1 / width,
        target.y1 / height,
        target.x2 / width,
        target.y2 / height,
    )
    for observation in observations:
        center_x = (observation.x1 + observation.x2) / 2
        center_y = (observation.y1 + observation.y2) / 2
        if not (
            normalized[0] <= center_x <= normalized[2]
            and normalized[1] <= center_y <= normalized[3]
        ):
            continue
        for identity in observation.identities:
            identities.setdefault(identity.identity, identity)
    return replace(target, identities=tuple(identities.values()))
