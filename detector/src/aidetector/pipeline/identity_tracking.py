from collections import defaultdict, deque
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Protocol

import numpy as np
from aidetector.domain.identity import (
    IdentityCandidate,
    IdentityMatch,
    TrackletSnapshot,
)
from aidetector.domain.vectors import normalize_vector
from numpy import ndarray


class IdentityMatcher(Protocol):
    def match(self, embedding: ndarray, /) -> IdentityMatch | None: ...


@dataclass
class _Track:
    track_id: int
    first_frame: int
    last_frame: int
    embeddings: deque[ndarray]
    embedding_sum: ndarray
    observations: int = 0
    identity: IdentityMatch | None = None
    sampling_stopped: bool = False


@dataclass(frozen=True, slots=True)
class IdentityUpdate:
    candidates: tuple[IdentityCandidate, ...]
    snapshots: tuple[TrackletSnapshot, ...]


class TrackIdentityAggregator:
    def __init__(
        self,
        gallery: IdentityMatcher | None,
        *,
        samples: int,
        max_age: int,
    ):
        if samples < 1 or max_age < 1:
            raise ValueError("Track samples and max age must be positive")
        self.gallery = gallery
        self.samples = samples
        self.max_age = max_age
        self._frame_by_source: defaultdict[str, int] = defaultdict(int)
        self._tracks_by_source: defaultdict[str, dict[int, _Track]] = defaultdict(dict)

    def set_gallery(self, gallery: IdentityMatcher | None) -> None:
        self.gallery = gallery

    def stop_sampling(self, source: str, track_id: int) -> None:
        track = self._tracks_by_source.get(source, {}).get(track_id)
        if track is not None:
            track.sampling_stopped = True

    def apply(
        self,
        source: str,
        candidates: Sequence[IdentityCandidate],
        embeddings: ndarray,
    ) -> IdentityUpdate:
        if len(candidates) != len(embeddings):
            raise ValueError("Identity candidates and embeddings differ in length")

        frame = self._frame_by_source[source] + 1
        self._frame_by_source[source] = frame
        tracks = self._tracks_by_source[source]
        expired = [
            track_id
            for track_id, track in tracks.items()
            if frame - track.last_frame > self.max_age
        ]
        for track_id in expired:
            del tracks[track_id]

        updated = []
        snapshots = []
        for candidate, embedding in zip(candidates, embeddings, strict=True):
            track_id = candidate.detection.track_id
            if track_id is None:
                updated.append(candidate)
                continue
            track = tracks.get(track_id)
            if track is None:
                track = _Track(
                    track_id,
                    frame,
                    frame,
                    deque(maxlen=self.samples),
                    np.zeros_like(embedding, dtype=np.float32),
                )
                tracks[track.track_id] = track

            normalized = normalize_vector(embedding)
            track.last_frame = frame
            track.embeddings.append(normalized)
            track.embedding_sum += normalized
            track.observations += 1
            if (
                len(track.embeddings) == self.samples
                and track.observations % self.samples == 0
                and self.gallery is not None
            ):
                track.identity = self.gallery.match(
                    normalize_vector(np.mean(track.embeddings, axis=0))
                )
            updated.append(
                replace(
                    candidate,
                    detection=replace(
                        candidate.detection,
                        identities=(track.identity.result,) if track.identity else (),
                    ),
                )
            )
            if track.observations % self.samples == 0 and not track.sampling_stopped:
                snapshots.append(
                    TrackletSnapshot(
                        source,
                        track.track_id,
                        track.first_frame,
                        track.last_frame,
                        track.observations,
                        normalize_vector(track.embedding_sum),
                        candidate.image,
                        track.identity.key if track.identity else None,
                    )
                )
        return IdentityUpdate(tuple(updated), tuple(snapshots))
