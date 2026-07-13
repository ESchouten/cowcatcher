from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Protocol

from aidetector.domain.detections import Observation
from aidetector.domain.frames import Frame, FrameBatch
from aidetector.domain.identity import IdentityCandidate, TrackletSnapshot
from aidetector.pipeline.identity_tracking import (
    IdentityMatcher,
    TrackIdentityAggregator,
)
from aidetector.pipeline.ports import ModelBatchResult, ModelRunner
from numpy import ndarray


class SamplingDecision(Enum):
    CONTINUE = "continue"
    STOP = "stop"


@dataclass(frozen=True, slots=True)
class IdentityBatch:
    source: str
    frames: list[Frame]
    candidates: tuple[IdentityCandidate, ...]
    model_result: ModelBatchResult | None = None
    detection_observation: Observation | None = None


class IdentityLocalizer(Protocol):
    def candidates(
        self,
        result: Any,
        frame: ndarray,
    ) -> list[IdentityCandidate]: ...


class IdentityCandidateSource(Protocol):
    def batches(self, batch: FrameBatch) -> list[IdentityBatch]: ...


class IdentityEncoder(Protocol):
    def embed(self, images: list[ndarray]) -> ndarray: ...


class IdentityCatalog(Protocol):
    def sync(self) -> IdentityMatcher | None: ...

    def record(self, snapshot: TrackletSnapshot) -> SamplingDecision: ...

    def close(self) -> None: ...


class IdentityProvider(Protocol):
    def start(self) -> None: ...

    def process(self, batch: FrameBatch) -> list[IdentityBatch]: ...

    def close(self) -> None: ...


class ModelIdentityCandidateSource:
    def __init__(
        self,
        runner: ModelRunner,
        localizer: IdentityLocalizer,
        *,
        reuse_for_detection: bool,
    ):
        self.runner = runner
        self.localizer = localizer
        self.reuse_for_detection = reuse_for_detection

    def batches(self, batch: FrameBatch) -> list[IdentityBatch]:
        batches = []
        for tracked in self.runner.track_sources(batch):
            frame = tracked.frames[-1]
            candidates = self.localizer.candidates(
                tracked.result,
                frame.require_image(),
            )
            observations = (
                self.runner.observations_from_result(tracked.result, [frame])
                if self.reuse_for_detection
                else None
            )
            batches.append(
                IdentityBatch(
                    tracked.source,
                    tracked.frames,
                    tuple(candidates),
                    tracked if self.reuse_for_detection else None,
                    observations[-1] if observations else None,
                )
            )
        return batches


class TrackedIdentityProvider:
    def __init__(
        self,
        *,
        candidates: IdentityCandidateSource,
        encoder: IdentityEncoder,
        catalog: IdentityCatalog,
        tracks: TrackIdentityAggregator,
    ):
        self.candidate_source = candidates
        self.encoder = encoder
        self.catalog = catalog
        self.tracks = tracks

    def start(self) -> None:
        self.tracks.set_gallery(self.catalog.sync())

    def process(self, batch: FrameBatch) -> list[IdentityBatch]:
        return [
            replace(
                item,
                candidates=self._identify(item.source, item.candidates),
            )
            for item in self.candidate_source.batches(batch)
        ]

    def close(self) -> None:
        self.catalog.close()

    def _identify(
        self,
        source: str,
        candidates: tuple[IdentityCandidate, ...],
    ) -> tuple[IdentityCandidate, ...]:
        self.tracks.set_gallery(self.catalog.sync())
        if not candidates:
            return ()

        embeddings = self.encoder.embed([candidate.image for candidate in candidates])
        update = self.tracks.apply(source, candidates, embeddings)
        for snapshot in update.snapshots:
            if self.catalog.record(snapshot) is SamplingDecision.STOP:
                self.tracks.stop_sampling(snapshot.source, snapshot.track_id)
        return update.candidates
