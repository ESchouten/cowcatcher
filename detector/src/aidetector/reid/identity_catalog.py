from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
from aidetector.domain.vectors import normalize_rows
from numpy import ndarray

SCHEMA_VERSION = 2
EVIDENCE_FRAMES = 2


class IdentityCatalogError(RuntimeError):
    """Base error for the shared identity catalog."""


class UnsupportedIdentitySchemaError(IdentityCatalogError):
    """Raised instead of attempting to migrate an old identity database."""


@dataclass(frozen=True, slots=True)
class CatalogControl:
    operator_revision: int
    active_gallery_version: int | None
    configuration_sha256: str | None
    encoder_key: str | None
    embedding_dimension: int | None


@dataclass(frozen=True, slots=True)
class GalleryIdentity:
    visual_identity_id: str
    official_id: str
    prototype: ndarray


@dataclass(frozen=True, slots=True)
class GallerySnapshot:
    gallery_version: int
    operator_revision: int
    identities: tuple[GalleryIdentity, ...]


@dataclass(frozen=True, slots=True)
class StoredEvidence:
    tracklet_id: str
    visual_identity_id: str


class IdentityCatalog:
    """SQLite identity catalog shared by the detector and web app."""

    def __init__(self, path: Path):
        path = path.resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(
            path,
            timeout=5,
            isolation_level=None,
            check_same_thread=False,
        )
        self.connection.row_factory = sqlite3.Row
        try:
            self.connection.executescript(
                """
                PRAGMA foreign_keys = ON;
                PRAGMA busy_timeout = 5000;
                PRAGMA journal_mode = WAL;
                """
            )
            self._initialize()
        except BaseException:
            self.connection.close()
            raise

    def _initialize(self) -> None:
        user_version = int(self.connection.execute("PRAGMA user_version").fetchone()[0])
        has_tables = self.connection.execute(
            """
            SELECT 1 FROM sqlite_schema
            WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
            LIMIT 1
            """
        ).fetchone()
        if user_version == 0 and has_tables is None:
            self.connection.executescript(_schema_sql())
            return
        if user_version != SCHEMA_VERSION:
            raise UnsupportedIdentitySchemaError(
                "Unsupported identity database schema "
                f"{user_version}; expected a fresh schema {SCHEMA_VERSION} database"
            )

    def control(self) -> CatalogControl:
        row = self.connection.execute(
            """
            SELECT operator_revision, active_gallery_version,
                   configuration_sha256, encoder_key, embedding_dimension
            FROM control
            WHERE singleton = 1
            """
        ).fetchone()
        if row is None:
            raise UnsupportedIdentitySchemaError("Identity control record is missing")
        return CatalogControl(
            operator_revision=int(row["operator_revision"]),
            active_gallery_version=(
                int(row["active_gallery_version"])
                if row["active_gallery_version"] is not None
                else None
            ),
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
        with self.transaction():
            before = self.control()
            if self._has_evidence() and (
                before.encoder_key not in (None, encoder_key)
                or before.embedding_dimension not in (None, embedding_dimension)
            ):
                raise IdentityCatalogError(
                    "Identity catalog contains embeddings from another encoder"
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
        observation_count: int,
        evidence_status: str = "eligible",
    ) -> StoredEvidence:
        vector = np.ascontiguousarray(embedding, dtype=np.float32)
        control = self.control()
        if control.embedding_dimension != vector.size:
            raise ValueError(
                "Evidence embedding dimension does not match the configured encoder"
            )

        tracklet_id = _stable_id("trk", run_id, source, str(track_id))
        captured = _format_time(captured_at)
        image_sha256 = hashlib.sha256(preview_jpeg).hexdigest()
        now = _utc_now()
        with self.transaction():
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

            self.connection.execute(
                """
                INSERT INTO tracklets (
                    tracklet_id, run_id, source, track_id,
                    first_captured_at, last_captured_at, observation_count,
                    evidence_status, preview_jpeg, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(tracklet_id) DO UPDATE SET
                    last_captured_at = excluded.last_captured_at,
                    observation_count = excluded.observation_count,
                    evidence_status = excluded.evidence_status,
                    preview_jpeg = excluded.preview_jpeg,
                    updated_at = excluded.updated_at
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
                    preview_jpeg,
                    now,
                    now,
                ),
            )

            self.connection.execute(
                """
                INSERT INTO evidence_frames (
                    evidence_id, tracklet_id, frame_index, captured_at,
                    image_sha256, preview_jpeg, embedding,
                    embedding_dimension, quality, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1.0, ?)
                ON CONFLICT(tracklet_id, frame_index) DO NOTHING
                """,
                (
                    f"evd_{uuid.uuid4().hex}",
                    tracklet_id,
                    frame_index,
                    captured,
                    image_sha256,
                    preview_jpeg,
                    vector.tobytes(),
                    vector.size,
                    now,
                ),
            )

            if assignment is None:
                self.connection.execute(
                    """
                    INSERT INTO visual_identity_tracklets (
                        tracklet_id, visual_identity_id, assignment_kind,
                        assigned_at
                    ) VALUES (?, ?, 'initial', ?)
                    """,
                    (tracklet_id, visual_identity_id, now),
                )
        return StoredEvidence(tracklet_id, visual_identity_id)

    def mark_tracklet_switch_risk(self, tracklet_id: str) -> None:
        self.connection.execute(
            """
            UPDATE tracklets
            SET evidence_status = 'switch_risk', updated_at = ?
            WHERE tracklet_id = ?
            """,
            (_utc_now(), tracklet_id),
        )

    def finalize_tracklet(
        self,
        tracklet_id: str,
        *,
        evidence_status: Literal["eligible", "insufficient"],
    ) -> None:
        now = _utc_now()
        with self.transaction():
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
                if frame_count < EVIDENCE_FRAMES:
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

    def gallery(self) -> GallerySnapshot:
        control = self.control()
        if control.active_gallery_version is None:
            return self.rebuild_gallery()
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
            or version["encoder_key"] != control.encoder_key
            or version["configuration_sha256"] != control.configuration_sha256
            or int(version["embedding_dimension"]) != control.embedding_dimension
        ):
            return self.rebuild_gallery()
        identities = tuple(
            GalleryIdentity(
                visual_identity_id=str(row["visual_identity_id"]),
                official_id=str(row["official_id"]),
                prototype=_embedding_from_blob(row["prototype"]),
            )
            for row in self.connection.execute(
                """
                SELECT visual_identity_id, official_id, prototype
                FROM gallery_items
                WHERE gallery_version = ?
                ORDER BY visual_identity_id
                """,
                (control.active_gallery_version,),
            )
        )
        return GallerySnapshot(
            int(version["gallery_version"]),
            control.operator_revision,
            identities,
        )

    def rebuild_gallery(self) -> GallerySnapshot:
        with self.transaction():
            control = self.control()
            if (
                control.encoder_key is None
                or control.configuration_sha256 is None
                or control.embedding_dimension is None
            ):
                raise IdentityCatalogError("Identity encoder is not configured")
            gallery_rows: list[tuple[str, str, ndarray, tuple[str, ...]]] = []
            for mapping in self.connection.execute(
                """
                SELECT visual_identity_id, official_id,
                       provisional_tracklet_id, confirmation_tracklet_id
                FROM mappings
                WHERE state = 'confirmed'
                ORDER BY visual_identity_id
                """
            ):
                selected: list[sqlite3.Row] = []
                for tracklet_id in (
                    str(mapping["provisional_tracklet_id"]),
                    str(mapping["confirmation_tracklet_id"]),
                ):
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
                            "Confirmed mapping references ineligible evidence"
                        )
                    frames = list(
                        self.connection.execute(
                            """
                            SELECT evidence_id, embedding
                            FROM evidence_frames
                            WHERE tracklet_id = ?
                            ORDER BY frame_index
                            LIMIT ?
                            """,
                            (tracklet_id, EVIDENCE_FRAMES),
                        )
                    )
                    if len(frames) != EVIDENCE_FRAMES:
                        raise IdentityCatalogError(
                            f"Gallery identities need {EVIDENCE_FRAMES} frames "
                            "from both visits"
                        )
                    selected.extend(frames)
                prototype = _normalized_mean(
                    [_embedding_from_blob(row["embedding"]) for row in selected]
                )
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
                    control.operator_revision,
                    control.encoder_key,
                    control.configuration_sha256,
                    content_sha256,
                    control.embedding_dimension,
                    now,
                ),
            )
            gallery_version = cursor.lastrowid
            assert gallery_version is not None
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
                        control.embedding_dimension,
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
                SET active_gallery_version = ?, updated_at = ?
                WHERE singleton = 1
                """,
                (gallery_version, now),
            )
        return GallerySnapshot(
            gallery_version,
            control.operator_revision,
            tuple(
                GalleryIdentity(visual_identity_id, official_id, prototype)
                for visual_identity_id, official_id, prototype, _ in gallery_rows
            ),
        )

    def close(self) -> None:
        self.connection.close()

    @contextmanager
    def transaction(self) -> Iterator[None]:
        self.connection.execute("BEGIN IMMEDIATE")
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

    def __enter__(self) -> IdentityCatalog:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def _schema_sql() -> str:
    return Path(__file__).with_name("schema.sql").read_text(encoding="utf-8")


def _embedding_from_blob(blob: bytes) -> ndarray:
    return np.frombuffer(blob, dtype=np.float32).copy()


def _normalized_mean(vectors: Sequence[ndarray]) -> ndarray:
    return normalize_rows(np.mean(np.stack(vectors), axis=0, keepdims=True))[0]


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
        digest.update(json.dumps(evidence_ids, separators=(",", ":")).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:32]}"


def _format_time(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
