from collections import defaultdict, deque
from dataclasses import dataclass, replace

import numpy as np
from aidetector.dazzlecow.gallery import IdentityMatch, IdentityMatcher
from aidetector.dazzlecow.localizer import CowCandidate
from numpy import ndarray


@dataclass
class _Track:
    track_id: int
    first_frame: int
    last_frame: int
    embeddings: deque[ndarray]
    embedding_sum: ndarray
    observations: int = 0
    identity: IdentityMatch | None = None
    stored: bool = False


@dataclass(frozen=True)
class TrackletSnapshot:
    source: str
    track_id: int
    first_frame: int
    last_frame: int
    observations: int
    embedding: ndarray
    preview: ndarray
    identity_key: str | None = None


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

    def set_gallery(self, gallery: IdentityMatcher) -> None:
        self.gallery = gallery
        for tracks in self._tracks_by_source.values():
            for track in tracks.values():
                track.identity = None

    def mark_stored(self, source: str, track_id: int) -> None:
        track = self._tracks_by_source[source].get(track_id)
        if track is not None:
            track.stored = True

    def apply(
        self,
        source: str,
        candidates: list[CowCandidate],
        embeddings: ndarray,
    ) -> list[TrackletSnapshot]:
        if len(candidates) != len(embeddings):
            raise ValueError("Cow candidates and embeddings differ in length")

        frame = self._frame_by_source[source] + 1
        self._frame_by_source[source] = frame
        tracks = self._tracks_by_source[source]
        for track_id in [
            track_id
            for track_id, track in tracks.items()
            if frame - track.last_frame > self.max_age
        ]:
            del tracks[track_id]

        snapshots = []
        for candidate, embedding in zip(candidates, embeddings, strict=True):
            track_id = candidate.crop.track_id
            if track_id is None:
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

            embedding = _normalize(embedding)
            track.last_frame = frame
            track.embeddings.append(embedding)
            track.embedding_sum += embedding
            track.observations += 1
            if (
                len(track.embeddings) == self.samples
                and track.observations % self.samples == 0
                and self.gallery is not None
            ):
                track.identity = self.gallery.match(
                    _normalize(np.mean(track.embeddings, axis=0))
                )
            candidate.crop = replace(
                candidate.crop,
                identities=(track.identity.result,) if track.identity else (),
            )
            if track.observations % self.samples == 0 and not track.stored:
                snapshots.append(
                    TrackletSnapshot(
                        source,
                        track.track_id,
                        track.first_frame,
                        track.last_frame,
                        track.observations,
                        _normalize(track.embedding_sum),
                        candidate.image,
                        track.identity.key if track.identity else None,
                    )
                )
        return snapshots


def _normalize(embedding: ndarray) -> ndarray:
    embedding = np.asarray(embedding, dtype=np.float32)
    norm = np.linalg.norm(embedding)
    return embedding / max(float(norm), np.finfo(np.float32).eps)
