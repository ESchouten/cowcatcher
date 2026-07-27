from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy import ndarray

SCHEMA_VERSION = 2
DEFAULT_BUSY_TIMEOUT_MS = 5_000
REQUIRED_TABLES = frozenset(
    {
        "audit_events",
        "control",
        "evidence_frames",
        "gallery_items",
        "gallery_versions",
        "mappings",
        "official_identities",
        "tracklets",
        "visual_identities",
        "visual_identity_tracklets",
    }
)


class IdentityCatalogError(RuntimeError):
    """Base error for the shared identity catalog."""


class UnsupportedIdentitySchemaError(IdentityCatalogError):
    """Raised instead of attempting to migrate an old identity database."""


class IdentityRevisionError(IdentityCatalogError):
    """Raised when an operation would use stale gallery/operator state."""


class CorruptIdentityBlobError(IdentityCatalogError):
    """Raised when a stored embedding is malformed or no longer normalized."""


@dataclass(frozen=True, slots=True)
class CatalogControl:
    schema_version: int
    operator_revision: int
    runtime_revision: int
    active_gallery_version: int | None
    last_identity_error: str | None
    configuration_sha256: str | None
    encoder_key: str | None
    embedding_dimension: int | None


@dataclass(frozen=True, slots=True)
class GalleryIdentity:
    visual_identity_id: str
    official_id: str
    prototype: ndarray
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GallerySnapshot:
    gallery_version: int | None
    operator_revision: int
    encoder_key: str
    configuration_sha256: str
    embedding_dimension: int
    identities: tuple[GalleryIdentity, ...]


@dataclass(frozen=True, slots=True)
class StoredEvidence:
    tracklet_id: str
    visual_identity_id: str
    evidence_id: str


class IdentityCatalog:
    """Python-owned SQLite catalog shared with the SvelteKit server.

    Opening the catalog initializes only a genuinely new database. Any existing
    non-empty database with another schema version is rejected deliberately;
    identity data was not previously deployed and no compatibility path exists.
    """

    def __init__(
        self,
        path: Path,
        *,
        busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS,
    ):
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        existed = self.path.exists()
        self.connection = sqlite3.connect(
            self.path,
            timeout=busy_timeout_ms / 1_000,
            isolation_level=None,
            check_same_thread=False,
        )
        self.connection.row_factory = sqlite3.Row
        try:
            self._configure_connection(busy_timeout_ms)
            self._initialize_or_validate(existed)
        except BaseException:
            self.connection.close()
            raise

    def _configure_connection(self, busy_timeout_ms: int) -> None:
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute(f"PRAGMA busy_timeout = {int(busy_timeout_ms)}")
        self.connection.execute("PRAGMA journal_mode = WAL")
        foreign_keys = int(self.connection.execute("PRAGMA foreign_keys").fetchone()[0])
        journal_mode = str(
            self.connection.execute("PRAGMA journal_mode").fetchone()[0]
        ).lower()
        if foreign_keys != 1 or journal_mode != "wal":
            raise IdentityCatalogError("Could not enable required SQLite pragmas")

    def _initialize_or_validate(self, existed: bool) -> None:
        user_version = int(self.connection.execute("PRAGMA user_version").fetchone()[0])
        tables = self._table_names()
        if user_version == 0 and not tables:
            self.connection.executescript(_schema_sql())
            self._validate_schema()
            return
        if not existed and (user_version != 0 or tables):
            raise IdentityCatalogError("New identity database was not empty")
        self._validate_schema()

    def _validate_schema(self) -> None:
        user_version = int(self.connection.execute("PRAGMA user_version").fetchone()[0])
        if user_version != SCHEMA_VERSION:
            raise UnsupportedIdentitySchemaError(
                "Unsupported identity database schema "
                f"{user_version}; expected a fresh schema {SCHEMA_VERSION} database"
            )
        missing = REQUIRED_TABLES - self._table_names()
        if missing:
            raise UnsupportedIdentitySchemaError(
                "Identity database is missing required tables: "
                + ", ".join(sorted(missing))
            )
        row = self.connection.execute(
            """
            SELECT schema_version
            FROM control
            WHERE singleton = 1
            """
        ).fetchone()
        if row is None or int(row["schema_version"]) != SCHEMA_VERSION:
            raise UnsupportedIdentitySchemaError(
                "Identity database control record does not match its schema"
            )
        violations = list(self.connection.execute("PRAGMA foreign_key_check"))
        if violations:
            raise UnsupportedIdentitySchemaError(
                "Identity database contains foreign-key violations"
            )

    def _table_names(self) -> set[str]:
        return {
            str(row["name"])
            for row in self.connection.execute(
                """
                SELECT name
                FROM sqlite_schema
                WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
                """
            )
        }

    def control(self) -> CatalogControl:
        row = self.connection.execute(
            """
            SELECT schema_version, operator_revision, runtime_revision,
                   active_gallery_version, last_identity_error,
                   configuration_sha256, encoder_key, embedding_dimension
            FROM control
            WHERE singleton = 1
            """
        ).fetchone()
        if row is None:
            raise UnsupportedIdentitySchemaError("Identity control record is missing")
        return CatalogControl(
            schema_version=int(row["schema_version"]),
            operator_revision=int(row["operator_revision"]),
            runtime_revision=int(row["runtime_revision"]),
            active_gallery_version=(
                int(row["active_gallery_version"])
                if row["active_gallery_version"] is not None
                else None
            ),
            last_identity_error=row["last_identity_error"],
            configuration_sha256=row["configuration_sha256"],
            encoder_key=row["encoder_key"],
            embedding_dimension=(
                int(row["embedding_dimension"])
                if row["embedding_dimension"] is not None
                else None
            ),
        )

    def configure_runtime(
        self,
        *,
        encoder_key: str,
        embedding_dimension: int,
        configuration_sha256: str,
    ) -> CatalogControl:
        if not encoder_key or embedding_dimension < 1:
            raise ValueError("Encoder key and embedding dimension are required")
        _validate_sha256(configuration_sha256, "configuration")
        with self.transaction(immediate=True):
            before = self.control()
            if (
                before.encoder_key is not None
                and before.encoder_key != encoder_key
                and self._has_evidence()
            ):
                raise IdentityCatalogError(
                    "Identity catalog contains evidence for another encoder"
                )
            if (
                before.embedding_dimension is not None
                and before.embedding_dimension != embedding_dimension
                and self._has_evidence()
            ):
                raise IdentityCatalogError(
                    "Identity catalog contains incompatible embeddings"
                )
            changed = (
                before.encoder_key != encoder_key
                or before.embedding_dimension != embedding_dimension
                or before.configuration_sha256 != configuration_sha256
            )
            if changed:
                now = _utc_now()
                self.connection.execute(
                    """
                    UPDATE control
                    SET encoder_key = ?,
                        embedding_dimension = ?,
                        configuration_sha256 = ?,
                        active_gallery_version = NULL,
                        last_identity_error = NULL,
                        runtime_revision = runtime_revision + 1,
                        updated_at = ?
                    WHERE singleton = 1
                    """,
                    (
                        encoder_key,
                        embedding_dimension,
                        configuration_sha256,
                        now,
                    ),
                )
                self._insert_audit_event(
                    event_type="runtime_configured",
                    actor="detector",
                    entity_type="identity_catalog",
                    entity_id="control",
                    before=_jsonable_control(before),
                    after=_jsonable_control(self.control()),
                    reason="resolved detector configuration",
                    operator_revision=None,
                    occurred_at=now,
                )
        return self.control()

    def record_evidence(
        self,
        *,
        run_id: str,
        source: str,
        track_id: int,
        frame_index: int,
        captured_at: datetime,
        preview_jpeg: bytes,
        embedding: ndarray,
        quality: float,
        observation_count: int,
        evidence_status: str = "eligible",
        predicted_visual_identity_id: str | None = None,
    ) -> StoredEvidence:
        if not run_id or not source or track_id < 0 or frame_index < 0:
            raise ValueError("Evidence requires a run, source, and non-negative IDs")
        if observation_count < 1:
            raise ValueError("Observation count must be positive")
        if evidence_status not in {
            "eligible",
            "insufficient",
            "impure",
            "switch_risk",
        }:
            raise ValueError(f"Unsupported evidence status: {evidence_status}")
        if not 0.0 <= quality <= 1.0 or not math.isfinite(quality):
            raise ValueError("Evidence quality must be finite and between zero and one")
        if not preview_jpeg:
            raise ValueError("Evidence preview must not be empty")
        vector = _validated_embedding(embedding)
        control = self.control()
        if control.embedding_dimension != vector.size:
            raise ValueError(
                "Evidence embedding dimension does not match the configured encoder"
            )

        tracklet_id = _stable_id("trk", run_id, source, str(track_id))
        captured = _format_time(captured_at)
        image_sha256 = hashlib.sha256(preview_jpeg).hexdigest()
        now = _utc_now()
        with self.transaction(immediate=True):
            existing = self.connection.execute(
                """
                SELECT first_captured_at
                FROM tracklets
                WHERE tracklet_id = ?
                """,
                (tracklet_id,),
            ).fetchone()
            assignment = self.connection.execute(
                """
                SELECT visual_identity_id
                FROM visual_identity_tracklets
                WHERE tracklet_id = ?
                """,
                (tracklet_id,),
            ).fetchone()
            if assignment is None:
                visual_identity_id = f"vid_{uuid.uuid4().hex}"
                self.connection.execute(
                    """
                    INSERT INTO visual_identities (
                        visual_identity_id, status, created_at, updated_at
                    ) VALUES (?, 'pending', ?, ?)
                    """,
                    (visual_identity_id, now, now),
                )
            else:
                visual_identity_id = str(assignment["visual_identity_id"])

            if existing is None:
                self.connection.execute(
                    """
                    INSERT INTO tracklets (
                        tracklet_id, run_id, source, track_id,
                        first_captured_at, last_captured_at, observation_count,
                        evidence_status, predicted_visual_identity_id,
                        preview_jpeg, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        tracklet_id,
                        run_id,
                        source,
                        track_id,
                        captured,
                        captured,
                        observation_count,
                        evidence_status,
                        predicted_visual_identity_id,
                        preview_jpeg,
                        now,
                        now,
                    ),
                )
            else:
                self.connection.execute(
                    """
                    UPDATE tracklets
                    SET last_captured_at = ?,
                        observation_count = ?,
                        evidence_status = ?,
                        predicted_visual_identity_id = ?,
                        preview_jpeg = ?,
                        updated_at = ?
                    WHERE tracklet_id = ?
                    """,
                    (
                        captured,
                        observation_count,
                        evidence_status,
                        predicted_visual_identity_id,
                        preview_jpeg,
                        now,
                        tracklet_id,
                    ),
                )

            evidence_id = f"evd_{uuid.uuid4().hex}"
            cursor = self.connection.execute(
                """
                INSERT INTO evidence_frames (
                    evidence_id, tracklet_id, frame_index, captured_at,
                    image_sha256, preview_jpeg, embedding,
                    embedding_dimension, quality, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(tracklet_id, frame_index) DO NOTHING
                """,
                (
                    evidence_id,
                    tracklet_id,
                    frame_index,
                    captured,
                    image_sha256,
                    preview_jpeg,
                    vector.tobytes(),
                    vector.size,
                    quality,
                    now,
                ),
            )
            if cursor.rowcount == 0:
                row = self.connection.execute(
                    """
                    SELECT evidence_id
                    FROM evidence_frames
                    WHERE tracklet_id = ? AND frame_index = ?
                    """,
                    (tracklet_id, frame_index),
                ).fetchone()
                evidence_id = str(row["evidence_id"])

            event_id = self._insert_audit_event(
                event_type=(
                    "tracklet_created" if existing is None else "tracklet_observed"
                ),
                actor="detector",
                entity_type="tracklet",
                entity_id=tracklet_id,
                before=None,
                after={
                    "visual_identity_id": visual_identity_id,
                    "evidence_id": evidence_id,
                    "frame_index": frame_index,
                    "observation_count": observation_count,
                    "predicted_visual_identity_id": predicted_visual_identity_id,
                },
                reason="eligible primary tracker observation",
                operator_revision=None,
                occurred_at=now,
            )
            if assignment is None:
                self.connection.execute(
                    """
                    INSERT INTO visual_identity_tracklets (
                        tracklet_id, visual_identity_id, assignment_kind,
                        audit_event_id, assigned_at
                    ) VALUES (?, ?, 'initial', ?, ?)
                    """,
                    (tracklet_id, visual_identity_id, event_id, now),
                )
            self.connection.execute(
                """
                UPDATE control
                SET runtime_revision = runtime_revision + 1,
                    updated_at = ?
                WHERE singleton = 1
                """,
                (now,),
            )
        return StoredEvidence(tracklet_id, visual_identity_id, evidence_id)

    def mark_tracklet_switch_risk(
        self,
        tracklet_id: str,
        *,
        reason: str,
    ) -> None:
        if not tracklet_id or not reason.strip():
            raise ValueError("Switch-risk updates require a tracklet and reason")
        now = _utc_now()
        with self.transaction(immediate=True):
            row = self.connection.execute(
                """
                SELECT evidence_status
                FROM tracklets
                WHERE tracklet_id = ?
                """,
                (tracklet_id,),
            ).fetchone()
            if row is None:
                raise IdentityCatalogError(
                    f"Cannot mark missing tracklet as switch risk: {tracklet_id}"
                )
            before = str(row["evidence_status"])
            if before == "switch_risk":
                return
            self.connection.execute(
                """
                UPDATE tracklets
                SET evidence_status = 'switch_risk', updated_at = ?
                WHERE tracklet_id = ?
                """,
                (now, tracklet_id),
            )
            self._insert_audit_event(
                event_type="tracklet_switch_risk",
                actor="detector",
                entity_type="tracklet",
                entity_id=tracklet_id,
                before={"evidence_status": before},
                after={"evidence_status": "switch_risk"},
                reason=reason,
                operator_revision=None,
                occurred_at=now,
            )
            self.connection.execute(
                """
                UPDATE control
                SET runtime_revision = runtime_revision + 1,
                    updated_at = ?
                WHERE singleton = 1
                """,
                (now,),
            )

    def finalize_tracklet(
        self,
        tracklet_id: str,
        *,
        evidence_status: Literal["eligible", "insufficient"],
        minimum_evidence_frames: int,
        reason: str,
    ) -> None:
        if not tracklet_id or minimum_evidence_frames < 1 or not reason.strip():
            raise ValueError(
                "Tracklet finalization requires a tracklet, evidence count, and reason"
            )
        now = _utc_now()
        with self.transaction(immediate=True):
            row = self.connection.execute(
                """
                SELECT evidence_status
                FROM tracklets
                WHERE tracklet_id = ?
                """,
                (tracklet_id,),
            ).fetchone()
            if row is None:
                raise IdentityCatalogError(
                    f"Cannot finalize missing tracklet: {tracklet_id}"
                )
            before = str(row["evidence_status"])
            frame_count = int(
                self.connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM evidence_frames
                    WHERE tracklet_id = ?
                    """,
                    (tracklet_id,),
                ).fetchone()[0]
            )
            if evidence_status == "eligible":
                if before == "switch_risk":
                    raise IdentityCatalogError(
                        "Switch-risk evidence cannot become eligible"
                    )
                if frame_count < minimum_evidence_frames:
                    raise IdentityCatalogError(
                        "Eligible tracklet does not contain enough evidence frames"
                    )
            after = "switch_risk" if before == "switch_risk" else evidence_status
            self.connection.execute(
                """
                UPDATE tracklets
                SET evidence_status = ?, updated_at = ?
                WHERE tracklet_id = ?
                """,
                (after, now, tracklet_id),
            )
            self._insert_audit_event(
                event_type="tracklet_finalized",
                actor="detector",
                entity_type="tracklet",
                entity_id=tracklet_id,
                before={"evidence_status": before},
                after={
                    "evidence_status": after,
                    "evidence_frame_count": frame_count,
                    "minimum_evidence_frames": minimum_evidence_frames,
                },
                reason=reason,
                operator_revision=None,
                occurred_at=now,
            )
            self.connection.execute(
                """
                UPDATE control
                SET runtime_revision = runtime_revision + 1,
                    updated_at = ?
                WHERE singleton = 1
                """,
                (now,),
            )

    def load_active_gallery(
        self,
        *,
        expected_operator_revision: int,
        encoder_key: str,
        configuration_sha256: str,
        embedding_dimension: int,
    ) -> GallerySnapshot:
        control = self.control()
        if control.operator_revision != expected_operator_revision:
            raise IdentityRevisionError(
                "Identity operator revision changed during gallery reload"
            )
        if (
            control.encoder_key != encoder_key
            or control.configuration_sha256 != configuration_sha256
            or control.embedding_dimension != embedding_dimension
        ):
            raise IdentityRevisionError(
                "Identity gallery does not match the resolved configuration"
            )
        if control.active_gallery_version is None:
            return GallerySnapshot(
                None,
                control.operator_revision,
                encoder_key,
                configuration_sha256,
                embedding_dimension,
                (),
            )
        version = self.connection.execute(
            """
            SELECT gallery_version, operator_revision, encoder_key,
                   configuration_sha256, embedding_dimension, state
            FROM gallery_versions
            WHERE gallery_version = ?
            """,
            (control.active_gallery_version,),
        ).fetchone()
        if (
            version is None
            or version["state"] != "active"
            or int(version["operator_revision"]) != control.operator_revision
            or version["encoder_key"] != encoder_key
            or version["configuration_sha256"] != configuration_sha256
            or int(version["embedding_dimension"]) != embedding_dimension
        ):
            raise IdentityRevisionError(
                "Active identity gallery has a stale revision or configuration"
            )
        identities: list[GalleryIdentity] = []
        for row in self.connection.execute(
            """
            SELECT visual_identity_id, official_id, prototype,
                   embedding_dimension, evidence_ids_json
            FROM gallery_items
            WHERE gallery_version = ?
            ORDER BY visual_identity_id
            """,
            (control.active_gallery_version,),
        ):
            dimension = int(row["embedding_dimension"])
            if dimension != embedding_dimension:
                raise CorruptIdentityBlobError(
                    "Gallery item embedding dimension is inconsistent"
                )
            prototype = _embedding_from_blob(row["prototype"], dimension)
            evidence_ids = json.loads(str(row["evidence_ids_json"]))
            if not isinstance(evidence_ids, list) or not all(
                isinstance(item, str) for item in evidence_ids
            ):
                raise CorruptIdentityBlobError(
                    "Gallery evidence references are malformed"
                )
            identities.append(
                GalleryIdentity(
                    visual_identity_id=str(row["visual_identity_id"]),
                    official_id=str(row["official_id"]),
                    prototype=prototype,
                    evidence_ids=tuple(evidence_ids),
                )
            )
        return GallerySnapshot(
            int(version["gallery_version"]),
            int(version["operator_revision"]),
            str(version["encoder_key"]),
            str(version["configuration_sha256"]),
            int(version["embedding_dimension"]),
            tuple(identities),
        )

    def rebuild_gallery(
        self,
        *,
        expected_operator_revision: int,
        encoder_key: str,
        configuration_sha256: str,
        embedding_dimension: int,
        gallery_frames: int,
    ) -> GallerySnapshot:
        if gallery_frames < 2:
            raise ValueError("Gallery requires at least two evidence frames")
        with self.transaction(immediate=True):
            control = self.control()
            if control.operator_revision != expected_operator_revision:
                raise IdentityRevisionError(
                    "Identity operator revision changed before gallery activation"
                )
            if (
                control.encoder_key != encoder_key
                or control.configuration_sha256 != configuration_sha256
                or control.embedding_dimension != embedding_dimension
            ):
                raise IdentityRevisionError(
                    "Cannot activate a gallery for another configuration"
                )
            gallery_rows: list[tuple[str, str, ndarray, tuple[str, ...]]] = []
            mappings = list(
                self.connection.execute(
                    """
                    SELECT visual_identity_id, official_id,
                           provisional_tracklet_id, confirmation_tracklet_id
                    FROM mappings
                    WHERE state = 'confirmed'
                    ORDER BY visual_identity_id
                    """
                )
            )
            for mapping in mappings:
                tracklet_ids = (
                    str(mapping["provisional_tracklet_id"]),
                    str(mapping["confirmation_tracklet_id"]),
                )
                if tracklet_ids[0] == tracklet_ids[1]:
                    raise IdentityCatalogError(
                        "Confirmed identity references the same tracklet twice"
                    )
                per_tracklet = gallery_frames // 2
                remainder = gallery_frames - per_tracklet * 2
                requested = (per_tracklet + remainder, per_tracklet)
                selected: list[sqlite3.Row] = []
                for tracklet_id, limit in zip(tracklet_ids, requested, strict=True):
                    assignment = self.connection.execute(
                        """
                        SELECT t.evidence_status, vit.visual_identity_id
                        FROM tracklets t
                        JOIN visual_identity_tracklets vit
                          ON vit.tracklet_id = t.tracklet_id
                        WHERE t.tracklet_id = ?
                        """,
                        (tracklet_id,),
                    ).fetchone()
                    if (
                        assignment is None
                        or assignment["evidence_status"] != "eligible"
                        or assignment["visual_identity_id"]
                        != mapping["visual_identity_id"]
                    ):
                        raise IdentityCatalogError(
                            "Confirmed mapping references an ineligible or "
                            "unassigned tracklet"
                        )
                    frames = list(
                        self.connection.execute(
                            """
                            SELECT evidence_id, embedding, embedding_dimension
                            FROM evidence_frames
                            WHERE tracklet_id = ?
                            ORDER BY frame_index
                            LIMIT ?
                            """,
                            (tracklet_id, limit),
                        )
                    )
                    if len(frames) != limit:
                        raise IdentityCatalogError(
                            "Gallery activation requires explicitly referenced "
                            f"{gallery_frames} frames across two tracklets"
                        )
                    selected.extend(frames)
                if len(selected) != gallery_frames:
                    raise IdentityCatalogError(
                        "Gallery evidence count does not match policy"
                    )
                vectors = [
                    _embedding_from_blob(
                        row["embedding"], int(row["embedding_dimension"])
                    )
                    for row in selected
                ]
                if any(vector.size != embedding_dimension for vector in vectors):
                    raise CorruptIdentityBlobError(
                        "Evidence embedding dimension is incompatible"
                    )
                prototype = _normalized_mean(vectors)
                gallery_rows.append(
                    (
                        str(mapping["visual_identity_id"]),
                        str(mapping["official_id"]),
                        prototype,
                        tuple(str(row["evidence_id"]) for row in selected),
                    )
                )

            content_sha256 = _gallery_content_sha256(gallery_rows)
            now = _utc_now()
            self.connection.execute(
                """
                UPDATE gallery_versions
                SET state = 'superseded'
                WHERE state = 'active'
                """
            )
            cursor = self.connection.execute(
                """
                INSERT INTO gallery_versions (
                    operator_revision, encoder_key, configuration_sha256,
                    content_sha256, embedding_dimension, state, created_at
                ) VALUES (?, ?, ?, ?, ?, 'active', ?)
                """,
                (
                    expected_operator_revision,
                    encoder_key,
                    configuration_sha256,
                    content_sha256,
                    embedding_dimension,
                    now,
                ),
            )
            if cursor.lastrowid is None:
                raise IdentityCatalogError("SQLite did not return a gallery version")
            gallery_version = cursor.lastrowid
            self.connection.executemany(
                """
                INSERT INTO gallery_items (
                    gallery_version, visual_identity_id, official_id,
                    prototype, embedding_dimension, evidence_ids_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        gallery_version,
                        visual_identity_id,
                        official_id,
                        prototype.tobytes(),
                        embedding_dimension,
                        json.dumps(evidence_ids, separators=(",", ":")),
                    )
                    for (
                        visual_identity_id,
                        official_id,
                        prototype,
                        evidence_ids,
                    ) in gallery_rows
                ],
            )
            self.connection.execute(
                """
                UPDATE control
                SET active_gallery_version = ?,
                    last_identity_error = NULL,
                    runtime_revision = runtime_revision + 1,
                    updated_at = ?
                WHERE singleton = 1
                """,
                (gallery_version, now),
            )
            self._insert_audit_event(
                event_type="gallery_activated",
                actor="detector",
                entity_type="gallery_version",
                entity_id=str(gallery_version),
                before=None,
                after={
                    "gallery_version": gallery_version,
                    "operator_revision": expected_operator_revision,
                    "identity_count": len(gallery_rows),
                    "content_sha256": content_sha256,
                },
                reason="operator revision validated",
                operator_revision=expected_operator_revision,
                occurred_at=now,
            )
        return self.load_active_gallery(
            expected_operator_revision=expected_operator_revision,
            encoder_key=encoder_key,
            configuration_sha256=configuration_sha256,
            embedding_dimension=embedding_dimension,
        )

    def record_runtime_error(self, message: str) -> None:
        message = message.strip()
        if not message:
            raise ValueError("Runtime error message must not be empty")
        with self.transaction(immediate=True):
            before = self.control().last_identity_error
            if before == message:
                return
            now = _utc_now()
            self.connection.execute(
                """
                UPDATE control
                SET active_gallery_version = NULL,
                    last_identity_error = ?,
                    runtime_revision = runtime_revision + 1,
                    updated_at = ?
                WHERE singleton = 1
                """,
                (message, now),
            )
            self._insert_audit_event(
                event_type="identity_runtime_error",
                actor="detector",
                entity_type="identity_catalog",
                entity_id="control",
                before={"last_identity_error": before},
                after={"last_identity_error": message},
                reason="identity output disabled",
                operator_revision=None,
                occurred_at=now,
            )

    def close(self) -> None:
        self.connection.close()

    @contextmanager
    def transaction(self, *, immediate: bool = False) -> Iterator[None]:
        self.connection.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
        try:
            yield
        except BaseException:
            self.connection.rollback()
            raise
        else:
            self.connection.commit()

    def _has_evidence(self) -> bool:
        return (
            self.connection.execute("SELECT 1 FROM evidence_frames LIMIT 1").fetchone()
            is not None
        )

    def _insert_audit_event(
        self,
        *,
        event_type: str,
        actor: str,
        entity_type: str,
        entity_id: str,
        before: dict[str, Any] | None,
        after: dict[str, Any] | None,
        reason: str,
        operator_revision: int | None,
        occurred_at: str,
    ) -> str:
        previous = self.connection.execute(
            "SELECT event_id FROM audit_events ORDER BY sequence DESC LIMIT 1"
        ).fetchone()
        event_id = f"evt_{uuid.uuid4().hex}"
        before_json = _canonical_json(before) if before is not None else None
        after_json = _canonical_json(after) if after is not None else None
        content = {
            "event_id": event_id,
            "event_type": event_type,
            "actor": actor,
            "entity_type": entity_type,
            "entity_id": entity_id,
            "operator_revision": operator_revision,
            "previous_event_id": previous["event_id"] if previous else None,
            "before_json": before_json,
            "after_json": after_json,
            "reason": reason,
            "occurred_at": occurred_at,
        }
        content_sha256 = hashlib.sha256(
            _canonical_json(content).encode("utf-8")
        ).hexdigest()
        self.connection.execute(
            """
            INSERT INTO audit_events (
                event_id, event_type, actor, entity_type, entity_id,
                operator_revision, previous_event_id, before_json, after_json,
                reason, occurred_at, content_sha256
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                event_type,
                actor,
                entity_type,
                entity_id,
                operator_revision,
                content["previous_event_id"],
                before_json,
                after_json,
                reason,
                occurred_at,
                content_sha256,
            ),
        )
        return event_id

    def __enter__(self) -> IdentityCatalog:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def _schema_sql() -> str:
    return Path(__file__).with_name("schema.sql").read_text(encoding="utf-8")


def _validated_embedding(embedding: ndarray) -> ndarray:
    vector = np.asarray(embedding, dtype=np.float32)
    if vector.ndim != 1 or vector.size == 0 or not np.all(np.isfinite(vector)):
        raise ValueError("Embedding must be a finite one-dimensional vector")
    norm = float(np.linalg.norm(vector.astype(np.float64)))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=2e-5):
        raise ValueError("Stored identity embeddings must be independently normalized")
    return np.ascontiguousarray(vector)


def _embedding_from_blob(blob: bytes, dimension: int) -> ndarray:
    if dimension < 1 or len(blob) != dimension * np.dtype(np.float32).itemsize:
        raise CorruptIdentityBlobError("Identity embedding BLOB has an invalid size")
    vector = np.frombuffer(blob, dtype=np.float32).copy()
    try:
        return _validated_embedding(vector)
    except ValueError as error:
        raise CorruptIdentityBlobError(str(error)) from error


def _normalized_mean(vectors: Sequence[ndarray]) -> ndarray:
    if not vectors:
        raise ValueError("Cannot build a prototype without evidence")
    mean = np.mean(
        np.asarray(vectors, dtype=np.float32).astype(np.float64),
        axis=0,
        keepdims=True,
    ).astype(np.float32)
    norm = float(np.linalg.norm(mean, axis=1, keepdims=True)[0, 0])
    if not math.isfinite(norm) or norm <= 0:
        raise CorruptIdentityBlobError("Gallery evidence has a zero or invalid mean")
    return np.ascontiguousarray(mean / norm, dtype=np.float32)[0]


def _gallery_content_sha256(
    rows: Sequence[tuple[str, str, ndarray, tuple[str, ...]]],
) -> str:
    digest = hashlib.sha256()
    for visual_identity_id, official_id, prototype, evidence_ids in rows:
        digest.update(visual_identity_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(official_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(prototype.tobytes())
        digest.update(b"\0")
        digest.update(_canonical_json(evidence_ids).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:32]}"


def _validate_sha256(value: str, label: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{label.capitalize()} checksum must be lowercase SHA-256")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _jsonable_control(control: CatalogControl) -> dict[str, Any]:
    return asdict(control)


def _format_time(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
