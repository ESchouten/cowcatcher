from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
from aidetector.domain.vectors import normalized_mean
from numpy import ndarray

SCHEMA_VERSION = 3
EVIDENCE_FRAMES = 2


class IdentityCatalogError(RuntimeError):
    """The identity catalog is invalid or incompatible with this runtime."""


class UnsupportedIdentitySchemaError(IdentityCatalogError):
    """The identity database needs to be recreated."""


@dataclass(frozen=True, slots=True)
class CatalogControl:
    operator_revision: int
    encoder_signature: str | None
    embedding_dimension: int | None


@dataclass(frozen=True, slots=True)
class GalleryIdentity:
    visual_identity_id: str
    official_id: str
    prototype: ndarray


@dataclass(frozen=True, slots=True)
class GallerySnapshot:
    operator_revision: int
    identities: tuple[GalleryIdentity, ...]


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
            SELECT operator_revision, encoder_signature, embedding_dimension
            FROM control
            WHERE singleton = 1
            """
        ).fetchone()
        if row is None:
            raise UnsupportedIdentitySchemaError("Identity control record is missing")
        return CatalogControl(
            operator_revision=int(row["operator_revision"]),
            encoder_signature=row["encoder_signature"],
            embedding_dimension=(
                int(row["embedding_dimension"])
                if row["embedding_dimension"] is not None
                else None
            ),
        )

    def configure_runtime(
        self,
        *,
        encoder_signature: str,
        embedding_dimension: int,
    ) -> CatalogControl:
        with self.transaction():
            control = self.control()
            has_evidence = (
                self.connection.execute(
                    "SELECT 1 FROM evidence_frames LIMIT 1"
                ).fetchone()
                is not None
            )
            incompatible = control.encoder_signature not in (
                None,
                encoder_signature,
            ) or control.embedding_dimension not in (None, embedding_dimension)
            if has_evidence and incompatible:
                raise IdentityCatalogError(
                    "Identity catalog contains embeddings from another encoder"
                )
            if (
                control.encoder_signature != encoder_signature
                or control.embedding_dimension != embedding_dimension
            ):
                self.connection.execute(
                    """
                    UPDATE control
                    SET encoder_signature = ?, embedding_dimension = ?
                    WHERE singleton = 1
                    """,
                    (encoder_signature, embedding_dimension),
                )
        return self.control()

    def record_evidence(
        self,
        *,
        tracklet_id: str,
        source: str,
        frame_index: int,
        captured_at: datetime,
        preview_jpeg: bytes,
        embedding: ndarray,
        evidence_status: str = "insufficient",
    ) -> str:
        vector = np.ascontiguousarray(embedding, dtype=np.float32)
        control = self.control()
        if control.embedding_dimension != vector.size:
            raise ValueError(
                "Evidence embedding dimension does not match the configured encoder"
            )

        with self.transaction():
            assignment = self.connection.execute(
                """
                SELECT visual_identity_id
                FROM visual_identity_tracklets
                WHERE tracklet_id = ?
                """,
                (tracklet_id,),
            ).fetchone()
            visual_identity_id = (
                str(assignment["visual_identity_id"])
                if assignment is not None
                else f"vid_{uuid.uuid4().hex}"
            )
            if assignment is None:
                self.connection.execute(
                    """
                    INSERT INTO visual_identities (visual_identity_id)
                    VALUES (?)
                    """,
                    (visual_identity_id,),
                )

            self.connection.execute(
                """
                INSERT INTO tracklets (
                    tracklet_id, source, last_captured_at,
                    evidence_status, preview_jpeg
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(tracklet_id) DO UPDATE SET
                    last_captured_at = excluded.last_captured_at,
                    evidence_status = excluded.evidence_status,
                    preview_jpeg = excluded.preview_jpeg
                """,
                (
                    tracklet_id,
                    source,
                    _format_time(captured_at),
                    evidence_status,
                    preview_jpeg,
                ),
            )
            self.connection.execute(
                """
                INSERT OR IGNORE INTO evidence_frames (
                    tracklet_id, frame_index, embedding
                ) VALUES (?, ?, ?)
                """,
                (tracklet_id, frame_index, vector.tobytes()),
            )
            if assignment is None:
                self.connection.execute(
                    """
                    INSERT INTO visual_identity_tracklets (
                        tracklet_id, visual_identity_id
                    ) VALUES (?, ?)
                    """,
                    (tracklet_id, visual_identity_id),
                )
        return visual_identity_id

    def mark_tracklet_switch_risk(self, tracklet_id: str) -> None:
        self.connection.execute(
            """
            UPDATE tracklets
            SET evidence_status = 'switch_risk'
            WHERE tracklet_id = ?
            """,
            (tracklet_id,),
        )

    def finalize_tracklet(
        self,
        tracklet_id: str,
        *,
        evidence_status: Literal["eligible", "insufficient"],
    ) -> None:
        with self.transaction():
            row = self.connection.execute(
                "SELECT evidence_status FROM tracklets WHERE tracklet_id = ?",
                (tracklet_id,),
            ).fetchone()
            if row is None:
                raise IdentityCatalogError(
                    f"Cannot finalize missing tracklet: {tracklet_id}"
                )
            before = str(row["evidence_status"])
            if evidence_status == "eligible":
                if before == "switch_risk":
                    raise IdentityCatalogError(
                        "Switch-risk evidence cannot become eligible"
                    )
                frames = int(
                    self.connection.execute(
                        """
                        SELECT COUNT(*) FROM evidence_frames
                        WHERE tracklet_id = ?
                        """,
                        (tracklet_id,),
                    ).fetchone()[0]
                )
                if frames != EVIDENCE_FRAMES:
                    raise IdentityCatalogError(
                        f"Eligible tracklets need {EVIDENCE_FRAMES} evidence frames"
                    )
            self.connection.execute(
                """
                UPDATE tracklets
                SET evidence_status = ?
                WHERE tracklet_id = ?
                """,
                (
                    "switch_risk" if before == "switch_risk" else evidence_status,
                    tracklet_id,
                ),
            )

    def gallery(self) -> GallerySnapshot:
        control = self.control()
        if control.encoder_signature is None or control.embedding_dimension is None:
            raise IdentityCatalogError("Identity encoder is not configured")

        identities = []
        for mapping in self.connection.execute(
            """
            SELECT visual_identity_id, official_id,
                   provisional_tracklet_id, confirmation_tracklet_id
            FROM mappings
            WHERE state = 'confirmed'
            ORDER BY visual_identity_id
            """
        ):
            embeddings = []
            for tracklet_id in (
                str(mapping["provisional_tracklet_id"]),
                str(mapping["confirmation_tracklet_id"]),
            ):
                rows = list(
                    self.connection.execute(
                        """
                        SELECT ef.embedding
                        FROM tracklets t
                        JOIN visual_identity_tracklets vit
                          ON vit.tracklet_id = t.tracklet_id
                        JOIN evidence_frames ef
                          ON ef.tracklet_id = t.tracklet_id
                        WHERE t.tracklet_id = ?
                          AND t.evidence_status = 'eligible'
                          AND vit.visual_identity_id = ?
                        ORDER BY ef.frame_index
                        """,
                        (tracklet_id, mapping["visual_identity_id"]),
                    )
                )
                if len(rows) != EVIDENCE_FRAMES:
                    raise IdentityCatalogError(
                        f"Gallery identities need {EVIDENCE_FRAMES} frames "
                        "from both visits"
                    )
                embeddings.extend(
                    _embedding_from_blob(
                        row["embedding"],
                        control.embedding_dimension,
                    )
                    for row in rows
                )
            identities.append(
                GalleryIdentity(
                    visual_identity_id=str(mapping["visual_identity_id"]),
                    official_id=str(mapping["official_id"]),
                    prototype=normalized_mean(np.stack(embeddings)),
                )
            )
        return GallerySnapshot(control.operator_revision, tuple(identities))

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

    def __enter__(self) -> IdentityCatalog:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def _schema_sql() -> str:
    return Path(__file__).with_name("schema.sql").read_text(encoding="utf-8")


def _embedding_from_blob(blob: bytes, dimension: int) -> ndarray:
    vector = np.frombuffer(blob, dtype=np.float32).copy()
    if vector.size != dimension:
        raise IdentityCatalogError("Identity evidence has an invalid embedding size")
    return vector


def _format_time(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
