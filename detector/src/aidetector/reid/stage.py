from __future__ import annotations

import uuid
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field, replace

import cv2
import numpy as np
from numpy import ndarray

from aidetector.domain.detections import DetectedObject, Observation
from aidetector.domain.identity import IdentityResult
from aidetector.domain.vectors import normalized_mean
from aidetector.reid.identity_catalog import (
    GalleryIdentity,
    GallerySnapshot,
    IdentityCatalog,
)
from aidetector.reid.miewid import (
    MODEL_SIGNATURE,
    MiewIdEncoder,
)
from aidetector.reid.zone import ZoneGate, ZoneVisit

PREVIEW_JPEG_QUALITY = 90
MIN_AREA_RATIO = 0.005
MAX_AREA_RATIO = 0.3
SIMILARITY_THRESHOLD = 0.75
SIMILARITY_MARGIN = 0.05
QUERY_FRAMES = 2
TRACK_MAX_AGE = 10


@dataclass(slots=True)
class _TrackState:
    tracklet_id: str
    last_frame: int
    observation_count: int = 0
    query_embeddings: list[ndarray] = field(default_factory=list)
    visual_identity_id: str | None = None
    result: IdentityResult | None = None
    last_matched_visual_identity_id: str | None = None
    switch_risk: bool = False


@dataclass(frozen=True, slots=True)
class GalleryScore:
    visual_identity_id: str
    official_id: str
    similarity: float
    margin: float


class IdentityStage:
    """Attach identity only to exact primary `(source, TrackId)` observations."""

    def __init__(
        self,
        *,
        target_label: str,
        margin: float,
        encoder: MiewIdEncoder,
        catalog: IdentityCatalog,
    ):
        self.target_label = target_label
        self.encoder = encoder
        self.catalog = catalog
        self._zone_gate = ZoneGate(margin)
        self._gallery: GallerySnapshot | None = None
        self._frame_by_source: defaultdict[str, int] = defaultdict(int)
        self._tracks: defaultdict[str, dict[int, _TrackState]] = defaultdict(dict)

    def start(self) -> None:
        self.catalog.configure_runtime(
            encoder_signature=MODEL_SIGNATURE,
            embedding_dimension=self.encoder.feature_dim,
        )
        self._reload_gallery()

    def enrich(self, source: str, observation: Observation) -> Observation:
        self._synchronize_gallery()
        return self._enrich(source, observation)

    def close(self) -> None:
        try:
            for tracks in self._tracks.values():
                for state in tracks.values():
                    self._abandon_tracklet(state)
                tracks.clear()
        finally:
            self.catalog.close()

    def _enrich(self, source: str, observation: Observation) -> Observation:
        frame_number = self._frame_by_source[source] + 1
        self._frame_by_source[source] = frame_number
        tracks = self._tracks[source]
        targets = [
            (index, item)
            for index, item in enumerate(observation.objects)
            if item.label == self.target_label
        ]
        image = observation.frame.require_image()
        zone = self._zone_gate.evaluate(
            source,
            targets,
            frame_width=image.shape[1],
            frame_height=image.shape[0],
        )
        for track_id in zone.tainted_track_ids:
            self._mark_track_switch_risk(tracks.get(track_id))
        if zone.closed_visit is not None:
            self._finalize_zone_visit(tracks, zone.closed_visit)
        for track_id in [
            track_id
            for track_id, state in tracks.items()
            if track_id != zone.active_track_id
            and frame_number - state.last_frame > TRACK_MAX_AGE
        ]:
            self._abandon_tracklet(tracks.pop(track_id))
        if not targets:
            return observation

        objects = list(observation.objects)
        candidates: list[tuple[int, DetectedObject, ndarray, ndarray]] = []
        for index, item in targets:
            zone_state = zone.target_states[index]
            if zone_state == "switch_risk":
                state = tracks.get(item.track_id) if item.track_id is not None else None
                objects[index] = replace(
                    item,
                    identity=IdentityResult(
                        status="switch_risk",
                        visual_identity_id=(
                            state.visual_identity_id if state is not None else None
                        ),
                    ),
                )
                continue
            if zone_state != "eligible" or item.track_id is None:
                objects[index] = replace(
                    item,
                    identity=IdentityResult(
                        status="insufficient_evidence",
                    ),
                )
                continue
            crop = _candidate_crop(image, item)
            if crop is None:
                objects[index] = replace(
                    item,
                    identity=IdentityResult(
                        status="insufficient_evidence",
                    ),
                )
                continue
            bgr, rgb = crop
            candidates.append((index, item, bgr, rgb))

        if not candidates:
            return replace(observation, objects=tuple(objects))

        embeddings = self.encoder.embed([rgb for _, _, _, rgb in candidates])
        for candidate, embedding in zip(candidates, embeddings, strict=True):
            index, item, bgr, _rgb = candidate
            assert item.track_id is not None
            state = tracks.get(item.track_id)
            if state is None:
                state = _TrackState(
                    tracklet_id=f"trk_{uuid.uuid4().hex}",
                    last_frame=frame_number,
                )
                tracks[item.track_id] = state
            state.last_frame = frame_number
            state.observation_count += 1
            state.query_embeddings.append(
                np.ascontiguousarray(embedding, dtype=np.float32)
            )
            score: GalleryScore | None = None
            if len(state.query_embeddings) == QUERY_FRAMES:
                query = normalized_mean(np.stack(state.query_embeddings))
                state.query_embeddings.clear()
                score = score_identity_query(query, self._gallery_identities())
                state.result = decide_identity_score(score)
                if state.result.status == "matched":
                    matched = state.result.visual_identity_id
                    if (
                        state.last_matched_visual_identity_id is not None
                        and state.last_matched_visual_identity_id != matched
                    ):
                        self._mark_track_switch_risk(state)
                    state.last_matched_visual_identity_id = matched
            if state.observation_count <= QUERY_FRAMES:
                state.visual_identity_id = self.catalog.record_evidence(
                    tracklet_id=state.tracklet_id,
                    source=source,
                    frame_index=state.observation_count - 1,
                    captured_at=observation.frame.captured_at,
                    preview_jpeg=_encode_preview(bgr),
                    embedding=embedding,
                    evidence_status=(
                        "switch_risk" if state.switch_risk else "insufficient"
                    ),
                )
            if state.switch_risk:
                result = IdentityResult(
                    status="switch_risk",
                    visual_identity_id=state.visual_identity_id,
                    similarity=(score.similarity if score is not None else None),
                    margin=score.margin if score is not None else None,
                )
            elif state.result is None:
                result = IdentityResult(
                    status="insufficient_evidence",
                    visual_identity_id=state.visual_identity_id,
                )
            elif state.result.status in {"unknown", "ambiguous"}:
                result = replace(
                    state.result,
                    visual_identity_id=state.visual_identity_id,
                )
            else:
                result = state.result
            objects[index] = replace(item, identity=result)
        return replace(observation, objects=tuple(objects))

    def _mark_track_switch_risk(
        self,
        state: _TrackState | None,
    ) -> None:
        if state is None:
            return
        state.switch_risk = True
        if state.visual_identity_id is not None:
            self.catalog.mark_tracklet_switch_risk(state.tracklet_id)

    def _finalize_zone_visit(
        self,
        tracks: dict[int, _TrackState],
        visit: ZoneVisit,
    ) -> None:
        state = tracks.get(visit.track_id)
        if state is None:
            return
        if visit.switch_risk:
            self._mark_track_switch_risk(state)
        if state.visual_identity_id is not None:
            eligible = (
                not visit.switch_risk
                and not state.switch_risk
                and state.observation_count >= QUERY_FRAMES
            )
            self.catalog.finalize_tracklet(
                state.tracklet_id,
                evidence_status="eligible" if eligible else "insufficient",
            )
        del tracks[visit.track_id]

    def _abandon_tracklet(
        self,
        state: _TrackState,
    ) -> None:
        if state.visual_identity_id is None:
            return
        self.catalog.finalize_tracklet(
            state.tracklet_id,
            evidence_status="insufficient",
        )

    def _gallery_identities(self) -> tuple[GalleryIdentity, ...]:
        return self._gallery.identities if self._gallery is not None else ()

    def _synchronize_gallery(self) -> None:
        control = self.catalog.control()
        if (
            self._gallery is None
            or self._gallery.operator_revision != control.operator_revision
        ):
            self._reload_gallery()

    def _reload_gallery(self) -> None:
        gallery = self.catalog.gallery()
        changed = (
            self._gallery is None
            or self._gallery.operator_revision != gallery.operator_revision
        )
        self._gallery = gallery
        if changed:
            for tracks in self._tracks.values():
                for state in tracks.values():
                    state.query_embeddings.clear()
                    state.result = None
                    state.last_matched_visual_identity_id = None


def score_identity_query(
    query: ndarray,
    identities: Sequence[GalleryIdentity],
) -> GalleryScore | None:
    if not identities:
        return None
    vector = np.asarray(query, dtype=np.float32)
    matrix = np.stack([item.prototype for item in identities])
    similarities = matrix @ vector
    order = np.argsort(-similarities, kind="stable")
    best = int(order[0])
    similarity = float(similarities[best])
    margin = similarity - float(similarities[int(order[1])]) if len(order) > 1 else 2.0
    identity = identities[best]
    return GalleryScore(
        identity.visual_identity_id,
        identity.official_id,
        similarity,
        margin,
    )


def decide_identity_score(
    score: GalleryScore | None,
) -> IdentityResult:
    if score is None:
        return IdentityResult(status="unknown")
    if score.similarity < SIMILARITY_THRESHOLD:
        status = "unknown"
    elif score.margin < SIMILARITY_MARGIN:
        status = "ambiguous"
    else:
        status = "matched"
    return IdentityResult(
        status=status,
        visual_identity_id=(score.visual_identity_id if status == "matched" else None),
        official_id=score.official_id if status == "matched" else None,
        similarity=score.similarity,
        margin=score.margin,
    )


def _candidate_crop(
    frame: ndarray,
    detection: DetectedObject,
) -> tuple[ndarray, ndarray] | None:
    height, width = frame.shape[:2]
    x1 = max(0, min(width, detection.x1))
    y1 = max(0, min(height, detection.y1))
    x2 = max(0, min(width, detection.x2))
    y2 = max(0, min(height, detection.y2))
    if x2 <= x1 or y2 <= y1:
        return None
    area_ratio = (x2 - x1) * (y2 - y1) / (width * height)
    if not MIN_AREA_RATIO <= area_ratio <= MAX_AREA_RATIO:
        return None
    bgr = np.ascontiguousarray(frame[y1:y2, x1:x2])
    rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return bgr, rgb


def _encode_preview(bgr: ndarray) -> bytes:
    success, encoded = cv2.imencode(
        ".jpg",
        bgr,
        (cv2.IMWRITE_JPEG_QUALITY, PREVIEW_JPEG_QUALITY),
    )
    if not success:
        raise ValueError("Could not encode identity evidence preview")
    return encoded.tobytes()
