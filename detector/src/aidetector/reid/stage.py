from __future__ import annotations

import hashlib
import json
import logging
import uuid
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from typing import Literal

import cv2
import numpy as np
from numpy import ndarray

from aidetector.domain.detections import DetectedObject, Observation
from aidetector.domain.identity import IdentityResult
from aidetector.reid.controlled_zone import (
    ControlledZoneGate,
    ControlledZonePolicy,
    ZoneVisitClosure,
)
from aidetector.reid.identity_catalog import (
    GalleryIdentity,
    GallerySnapshot,
    IdentityCatalog,
    IdentityRevisionError,
)
from aidetector.reid.miewid import MiewIdDualCropEncoder, normalized_prototype
from aidetector.reid.models import MODEL_REGISTRY

logger = logging.getLogger(__name__)
ELIGIBLE_EVIDENCE_QUALITY = 1.0
PREVIEW_JPEG_QUALITY = 90


@dataclass(frozen=True, slots=True)
class CandidateFilterPolicy:
    min_area_ratio: float
    max_area_ratio: float
    frame_edge_margin: float


@dataclass(frozen=True, slots=True)
class IdentityPolicy:
    target_label: str
    candidate_filter: CandidateFilterPolicy
    controlled_zone: ControlledZonePolicy
    encoder: Literal["miewid-dual-crop-v1"]
    similarity_threshold: float
    similarity_margin: float
    query_frames: int
    gallery_frames: int
    track_max_age: int

    def __post_init__(self) -> None:
        if not self.target_label:
            raise ValueError("Identity target label is required")
        if not (
            0
            <= self.candidate_filter.min_area_ratio
            <= self.candidate_filter.max_area_ratio
            <= 1
        ):
            raise ValueError("Identity candidate area policy is invalid")
        if not 0 <= self.candidate_filter.frame_edge_margin < 0.5:
            raise ValueError("Identity frame edge margin is invalid")
        if not 0 <= self.similarity_threshold <= 1:
            raise ValueError("Identity similarity threshold is invalid")
        if not 0 <= self.similarity_margin <= 1:
            raise ValueError("Identity similarity margin is invalid")
        if self.query_frames != 2 or self.gallery_frames != 4:
            raise ValueError(
                "miewid-dual-crop-v1 requires two query and four gallery frames"
            )
        if self.track_max_age < 1:
            raise ValueError("Identity track maximum age must be positive")


@dataclass(slots=True)
class _TrackState:
    tracklet_run_id: str
    zone_visit_number: int
    first_frame: int
    last_frame: int
    catalog_tracklet_id: str | None = None
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
        policy: IdentityPolicy,
        encoder: MiewIdDualCropEncoder,
        catalog: IdentityCatalog,
        process_run_id: str | None = None,
    ):
        self.policy = policy
        self.encoder = encoder
        self.catalog = catalog
        self.process_run_id = process_run_id or uuid.uuid4().hex
        self.configuration_sha256 = identity_configuration_sha256(policy)
        self._zone_gate = ControlledZoneGate(policy.controlled_zone)
        self._gallery: GallerySnapshot | None = None
        self._started = False
        self._disabled_operator_revision: int | None = None
        self._disabled_message: str | None = None
        self._frame_by_source: defaultdict[str, int] = defaultdict(int)
        self._tracks: defaultdict[str, dict[int, _TrackState]] = defaultdict(dict)

    @property
    def gallery_version(self) -> int | None:
        return self._gallery.gallery_version if self._gallery is not None else None

    def start(self) -> None:
        if self._started:
            return
        try:
            self.catalog.configure_runtime(
                encoder_key=self.policy.encoder,
                embedding_dimension=self.encoder.feature_dim,
                configuration_sha256=self.configuration_sha256,
            )
            self._reload_gallery()
        except Exception as error:
            self._disable_identity(error)
        self._started = True

    def enrich(self, source: str, observation: Observation) -> Observation:
        if not self._started:
            raise RuntimeError("Identity stage is not started")
        if self._identity_is_disabled():
            return self._error_observation(observation)
        try:
            self._synchronize_gallery()
            return self._enrich(source, observation)
        except Exception as error:
            self._disable_identity(error)
            return self._error_observation(observation)

    def close(self) -> None:
        try:
            for tracks in self._tracks.values():
                for state in tracks.values():
                    self._abandon_tracklet(
                        state,
                        reason="detector stopped before controlled-zone clearance",
                    )
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
            if item.label == self.policy.target_label
        ]
        image = observation.frame.require_image()
        zone = self._zone_gate.evaluate(
            source,
            targets,
            frame_width=image.shape[1],
            frame_height=image.shape[0],
        )
        for track_id in zone.tainted_track_ids:
            self._mark_track_switch_risk(
                tracks.get(track_id),
                reason="controlled zone detected ambiguous or replaced tracker evidence",
            )
        if zone.closed_visit is not None:
            self._finalize_zone_visit(tracks, zone.closed_visit)
        for track_id in [
            track_id
            for track_id, state in tracks.items()
            if track_id != zone.active_track_id
            and frame_number - state.last_frame > self.policy.track_max_age
        ]:
            self._abandon_tracklet(
                tracks.pop(track_id),
                reason="identity track expired without controlled-zone clearance",
            )
        if not targets:
            return observation

        objects = list(observation.objects)
        candidates: list[tuple[int, DetectedObject, ndarray, ndarray, int]] = []
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
                        gallery_version=self.gallery_version,
                    ),
                )
                continue
            if zone_state != "eligible" or item.track_id is None:
                objects[index] = replace(
                    item,
                    identity=IdentityResult(
                        status="insufficient_evidence",
                        gallery_version=self.gallery_version,
                    ),
                )
                continue
            if zone.visit_number is None:
                raise ValueError("Eligible controlled-zone target has no visit")
            crop = _candidate_crop(
                image,
                item,
                self.policy.candidate_filter,
            )
            if crop is None:
                objects[index] = replace(
                    item,
                    identity=IdentityResult(
                        status="insufficient_evidence",
                        gallery_version=self.gallery_version,
                    ),
                )
                continue
            bgr, rgb = crop
            candidates.append((index, item, bgr, rgb, zone.visit_number))

        if not candidates:
            return replace(observation, objects=tuple(objects))

        embeddings = self.encoder.embed([rgb for _, _, _, rgb, _ in candidates])
        if len(embeddings) != len(candidates):
            raise ValueError("Identity encoder returned the wrong row count")
        for candidate, embedding in zip(candidates, embeddings, strict=True):
            index, item, bgr, _rgb, visit_number = candidate
            assert item.track_id is not None
            state = tracks.get(item.track_id)
            if state is not None and state.zone_visit_number != visit_number:
                self._abandon_tracklet(
                    state,
                    reason="controlled-zone visit changed without a matching clearance",
                )
                del tracks[item.track_id]
                state = None
            if state is None:
                state = _TrackState(
                    tracklet_run_id=(f"{self.process_run_id}-{uuid.uuid4().hex[:12]}"),
                    zone_visit_number=visit_number,
                    first_frame=frame_number,
                    last_frame=frame_number,
                )
                tracks[item.track_id] = state
            state.last_frame = frame_number
            state.observation_count += 1
            state.query_embeddings.append(
                np.ascontiguousarray(embedding, dtype=np.float32)
            )
            score: GalleryScore | None = None
            if len(state.query_embeddings) == self.policy.query_frames:
                query = normalized_prototype(np.stack(state.query_embeddings))
                state.query_embeddings.clear()
                score = score_identity_query(query, self._gallery_identities())
                state.result = decide_identity_score(
                    score,
                    similarity_threshold=self.policy.similarity_threshold,
                    similarity_margin=self.policy.similarity_margin,
                    gallery_version=self.gallery_version,
                )
                if state.result.status == "matched":
                    matched = state.result.visual_identity_id
                    if (
                        state.last_matched_visual_identity_id is not None
                        and state.last_matched_visual_identity_id != matched
                    ):
                        state.switch_risk = True
                    state.last_matched_visual_identity_id = matched
            predicted_visual_identity_id = (
                score.visual_identity_id
                if score is not None
                else (
                    state.result.visual_identity_id
                    if state.result is not None and state.result.status == "matched"
                    else None
                )
            )
            preview = _encode_preview(bgr)
            stored = self.catalog.record_evidence(
                run_id=state.tracklet_run_id,
                source=source,
                track_id=item.track_id,
                frame_index=min(
                    state.observation_count - 1,
                    self.policy.query_frames - 1,
                ),
                captured_at=observation.frame.captured_at,
                preview_jpeg=preview,
                embedding=embedding,
                quality=ELIGIBLE_EVIDENCE_QUALITY,
                observation_count=state.observation_count,
                evidence_status=(
                    "switch_risk" if state.switch_risk else "insufficient"
                ),
                predicted_visual_identity_id=predicted_visual_identity_id,
            )
            state.catalog_tracklet_id = stored.tracklet_id
            state.visual_identity_id = stored.visual_identity_id
            if state.switch_risk:
                result = IdentityResult(
                    status="switch_risk",
                    visual_identity_id=state.visual_identity_id,
                    similarity=(score.similarity if score is not None else None),
                    margin=score.margin if score is not None else None,
                    gallery_version=self.gallery_version,
                )
            elif state.result is None:
                result = IdentityResult(
                    status="insufficient_evidence",
                    visual_identity_id=state.visual_identity_id,
                    gallery_version=self.gallery_version,
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
        *,
        reason: str,
    ) -> None:
        if state is None:
            return
        state.switch_risk = True
        if state.catalog_tracklet_id is not None:
            self.catalog.mark_tracklet_switch_risk(
                state.catalog_tracklet_id,
                reason=reason,
            )

    def _finalize_zone_visit(
        self,
        tracks: dict[int, _TrackState],
        visit: ZoneVisitClosure,
    ) -> None:
        state = tracks.get(visit.track_id)
        if state is None or state.zone_visit_number != visit.visit_number:
            return
        if visit.switch_risk:
            self._mark_track_switch_risk(
                state,
                reason="controlled zone visit ended after tracker ambiguity",
            )
        if state.catalog_tracklet_id is not None:
            eligible = (
                not visit.switch_risk
                and not state.switch_risk
                and state.observation_count >= self.policy.query_frames
            )
            self.catalog.finalize_tracklet(
                state.catalog_tracklet_id,
                evidence_status="eligible" if eligible else "insufficient",
                minimum_evidence_frames=self.policy.query_frames,
                reason=(
                    "controlled zone visit cleared cleanly"
                    if eligible
                    else "controlled zone visit ended without eligible evidence"
                ),
            )
        del tracks[visit.track_id]

    def _abandon_tracklet(
        self,
        state: _TrackState,
        *,
        reason: str,
    ) -> None:
        if state.catalog_tracklet_id is None:
            return
        self.catalog.finalize_tracklet(
            state.catalog_tracklet_id,
            evidence_status="insufficient",
            minimum_evidence_frames=self.policy.query_frames,
            reason=reason,
        )

    def _gallery_identities(self) -> tuple[GalleryIdentity, ...]:
        return self._gallery.identities if self._gallery is not None else ()

    def _synchronize_gallery(self) -> None:
        control = self.catalog.control()
        if (
            self._gallery is None
            or self._gallery.operator_revision != control.operator_revision
            or self._gallery.gallery_version != control.active_gallery_version
            or control.last_identity_error is not None
        ):
            self._reload_gallery()

    def _reload_gallery(self) -> None:
        control = self.catalog.control()
        try:
            gallery = self.catalog.load_active_gallery(
                expected_operator_revision=control.operator_revision,
                encoder_key=self.policy.encoder,
                configuration_sha256=self.configuration_sha256,
                embedding_dimension=self.encoder.feature_dim,
            )
            if gallery.gallery_version is None:
                gallery = self.catalog.rebuild_gallery(
                    expected_operator_revision=control.operator_revision,
                    encoder_key=self.policy.encoder,
                    configuration_sha256=self.configuration_sha256,
                    embedding_dimension=self.encoder.feature_dim,
                    gallery_frames=self.policy.gallery_frames,
                )
        except IdentityRevisionError:
            gallery = self.catalog.rebuild_gallery(
                expected_operator_revision=control.operator_revision,
                encoder_key=self.policy.encoder,
                configuration_sha256=self.configuration_sha256,
                embedding_dimension=self.encoder.feature_dim,
                gallery_frames=self.policy.gallery_frames,
            )
        changed = (
            self._gallery is None
            or self._gallery.gallery_version != gallery.gallery_version
        )
        self._gallery = gallery
        self._disabled_operator_revision = None
        self._disabled_message = None
        if changed:
            for tracks in self._tracks.values():
                for state in tracks.values():
                    state.query_embeddings.clear()
                    state.result = None
                    state.last_matched_visual_identity_id = None

    def _disable_identity(self, error: Exception) -> None:
        self._gallery = None
        message = f"{type(error).__name__}: {error}"
        try:
            self._disabled_operator_revision = self.catalog.control().operator_revision
        except Exception:
            self._disabled_operator_revision = None
        self._disabled_message = message
        logger.exception("Identity output disabled: %s", message)
        try:
            self.catalog.record_runtime_error(message)
        except Exception:
            logger.exception("Could not persist the identity runtime error")

    def _identity_is_disabled(self) -> bool:
        if self._disabled_message is None:
            return False
        try:
            revision = self.catalog.control().operator_revision
        except Exception:
            return True
        if revision == self._disabled_operator_revision:
            return True
        self._disabled_message = None
        self._disabled_operator_revision = None
        return False

    def _error_observation(self, observation: Observation) -> Observation:
        return replace(
            observation,
            objects=tuple(
                replace(
                    item,
                    identity=IdentityResult(
                        status="error",
                        gallery_version=self.gallery_version,
                    ),
                )
                if item.label == self.policy.target_label
                else item
                for item in observation.objects
            ),
        )


def score_identity_query(
    query: ndarray,
    identities: Sequence[GalleryIdentity],
) -> GalleryScore | None:
    if not identities:
        return None
    vector = np.asarray(query, dtype=np.float32)
    if vector.shape != (identities[0].prototype.size,):
        raise ValueError("Identity query and gallery dimensions differ")
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
    *,
    similarity_threshold: float,
    similarity_margin: float,
    gallery_version: int | None,
) -> IdentityResult:
    if score is None:
        return IdentityResult(
            status="unknown",
            gallery_version=gallery_version,
        )
    if score.similarity < similarity_threshold:
        status = "unknown"
    elif score.margin < similarity_margin:
        status = "ambiguous"
    else:
        status = "matched"
    return IdentityResult(
        status=status,
        visual_identity_id=(score.visual_identity_id if status == "matched" else None),
        official_id=score.official_id if status == "matched" else None,
        similarity=score.similarity,
        margin=score.margin,
        gallery_version=gallery_version,
    )


def identity_configuration_sha256(policy: IdentityPolicy) -> str:
    model = MODEL_REGISTRY[policy.encoder]
    payload = {
        "policy": asdict(policy),
        "model": {
            "key": model.key,
            "model_version_id": model.model_version_id,
            "sha256": model.sha256,
            "size_bytes": model.size_bytes,
            "immutable_revision": model.immutable_revision,
            "state_key_count": model.state_key_count,
            "input_size": model.input_size,
            "embedding_dimension": model.embedding_dimension,
            "runtime_backend": model.runtime_backend,
            "device_order": model.device_order,
            "device_dtypes": model.device_dtypes,
            "preprocessing": asdict(model.preprocessing),
        },
    }
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _candidate_crop(
    frame: ndarray,
    detection: DetectedObject,
    policy: CandidateFilterPolicy,
) -> tuple[ndarray, ndarray] | None:
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
        raise ValueError("Identity stage requires BGR uint8 detector frames")
    height, width = frame.shape[:2]
    x1 = max(0, min(width, detection.x1))
    y1 = max(0, min(height, detection.y1))
    x2 = max(0, min(width, detection.x2))
    y2 = max(0, min(height, detection.y2))
    if x2 <= x1 or y2 <= y1:
        return None
    area_ratio = (x2 - x1) * (y2 - y1) / (width * height)
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    margin = policy.frame_edge_margin
    if not (
        policy.min_area_ratio <= area_ratio <= policy.max_area_ratio
        and width * margin <= center_x <= width * (1 - margin)
        and height * margin <= center_y <= height * (1 - margin)
    ):
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
