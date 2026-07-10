from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np
from aidetector.dazzlecow.gallery import DazzleCowGallery
from aidetector.dazzlecow.geometry import box_iou
from aidetector.dazzlecow.localizer import CowCandidate
from numpy import ndarray


@dataclass
class _Track:
    track_id: int
    box: tuple[int, int, int, int]
    last_frame: int
    embeddings: deque[ndarray]


class TrackIdentityAggregator:
    def __init__(
        self,
        gallery: DazzleCowGallery,
        *,
        samples: int,
        iou_threshold: float,
        max_age: int,
    ):
        if samples < 1 or max_age < 1:
            raise ValueError("Track samples and max age must be positive")
        if not 0 <= iou_threshold <= 1:
            raise ValueError("Track IoU must be between 0 and 1")
        self.gallery = gallery
        self.samples = samples
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self._frame_by_source: defaultdict[str, int] = defaultdict(int)
        self._next_id_by_source: defaultdict[str, int] = defaultdict(int)
        self._tracks_by_source: defaultdict[str, dict[int, _Track]] = defaultdict(dict)

    def apply(
        self,
        source: str,
        candidates: list[CowCandidate],
        embeddings: ndarray,
    ) -> None:
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

        assignments = self._assign(candidates, tracks)
        for index, (candidate, embedding) in enumerate(
            zip(candidates, embeddings, strict=True)
        ):
            track = assignments.get(index)
            if track is None:
                self._next_id_by_source[source] += 1
                track = _Track(
                    self._next_id_by_source[source],
                    _box(candidate),
                    frame,
                    deque(maxlen=self.samples),
                )
                tracks[track.track_id] = track

            track.box = _box(candidate)
            track.last_frame = frame
            track.embeddings.append(_normalize(embedding))
            candidate.crop.track_id = track.track_id
            if len(track.embeddings) == self.samples:
                candidate.crop.identity = self.gallery.match(
                    _normalize(np.mean(track.embeddings, axis=0))
                )

    def _assign(
        self,
        candidates: list[CowCandidate],
        tracks: dict[int, _Track],
    ) -> dict[int, _Track]:
        pairs = sorted(
            (
                (box_iou(_box(candidate), track.box), index, track)
                for index, candidate in enumerate(candidates)
                for track in tracks.values()
            ),
            reverse=True,
            key=lambda item: item[0],
        )
        assignments = {}
        used_tracks = set()
        for score, index, track in pairs:
            if score < self.iou_threshold:
                break
            if index in assignments or track.track_id in used_tracks:
                continue
            assignments[index] = track
            used_tracks.add(track.track_id)
        return assignments


def _box(candidate: CowCandidate) -> tuple[int, int, int, int]:
    crop = candidate.crop
    return crop.x1, crop.y1, crop.x2, crop.y2
def _normalize(embedding: ndarray) -> ndarray:
    embedding = np.asarray(embedding, dtype=np.float32)
    norm = np.linalg.norm(embedding)
    return embedding / max(float(norm), np.finfo(np.float32).eps)
