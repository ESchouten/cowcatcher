from collections.abc import Callable
from dataclasses import dataclass, replace
from threading import Lock
from time import monotonic

from aidetector.domain.detections import DetectedObject, Observation
from aidetector.domain.frames import FrameBatch
from aidetector.domain.identity import IdentityResult
from aidetector.pipeline.identity_provider import IdentityProvider
from aidetector.pipeline.ports import EnrichmentBatch


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


class IdentityEnricher:
    def __init__(
        self,
        registry: IdentityRegistry,
        producer: str,
        provider: IdentityProvider | None,
    ):
        self.registry = registry
        self.producer = producer
        self.provider = provider

    def start(self) -> None:
        if self.provider is not None:
            self.provider.start()

    def close(self) -> None:
        if self.provider is not None:
            self.provider.close()

    def process(self, batch: FrameBatch) -> EnrichmentBatch:
        provider = self.provider
        if provider is None:
            return EnrichmentBatch()

        observations = []
        identified = provider.process(batch)
        for item in identified:
            frame = item.frames[-1]
            identity_observation = Observation(
                frame,
                tuple(candidate.detection for candidate in item.candidates),
            )
            self.registry.publish(item.source, self.producer, identity_observation)
            observations.append(
                (
                    item.source,
                    self.registry.enrich(item.source, item.detection_observation)
                    if item.detection_observation is not None
                    else identity_observation,
                )
            )

        model_results = tuple(
            item.model_result for item in identified if item.model_result is not None
        )
        return EnrichmentBatch(
            tuple(observations),
            model_results or None,
        )

    def enrich(self, source: str, observation: Observation) -> Observation:
        return self.registry.enrich(source, observation)


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
