from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from aidetector.reid.identity_catalog import (
    REQUIRED_TABLES,
    SCHEMA_VERSION,
    CorruptIdentityBlobError,
    IdentityCatalog,
    IdentityCatalogError,
    UnsupportedIdentitySchemaError,
)


def normalized(*values: float) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    return vector / np.linalg.norm(vector.astype(np.float64))


def configure(catalog: IdentityCatalog, *, dimension: int = 2):
    return catalog.configure_runtime(
        encoder_key="miewid-dual-crop-v1",
        embedding_dimension=dimension,
        configuration_sha256="a" * 64,
    )


def test_python_initializes_the_complete_versioned_schema(tmp_path: Path):
    path = tmp_path / "identities" / "catalog.sqlite"

    with IdentityCatalog(path) as catalog:
        tables = {
            row["name"]
            for row in catalog.connection.execute(
                """
                SELECT name FROM sqlite_schema
                WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
                """
            )
        }
        assert REQUIRED_TABLES <= tables
        assert catalog.connection.execute("PRAGMA user_version").fetchone()[0] == (
            SCHEMA_VERSION
        )
        assert catalog.connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert catalog.connection.execute("PRAGMA journal_mode").fetchone()[0] == (
            "wal"
        )
        assert catalog.connection.execute("PRAGMA busy_timeout").fetchone()[0] == (
            5_000
        )

    assert path.is_file()


def test_existing_legacy_schema_is_rejected_without_mutation(tmp_path: Path):
    path = tmp_path / "legacy.sqlite"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE identities (identity TEXT PRIMARY KEY)")
    connection.commit()
    connection.close()

    with pytest.raises(UnsupportedIdentitySchemaError, match="Unsupported"):
        IdentityCatalog(path)

    connection = sqlite3.connect(path)
    assert connection.execute("PRAGMA user_version").fetchone()[0] == 0
    assert connection.execute(
        "SELECT name FROM sqlite_schema WHERE type = 'table'"
    ).fetchall() == [("identities",)]
    connection.close()


def test_pre_zone_schema_version_is_rejected_without_migration(tmp_path: Path):
    path = tmp_path / "version-1.sqlite"
    schema_path = Path(__file__).resolve().parents[1] / "src/aidetector/reid/schema.sql"
    version_1_schema = (
        schema_path.read_text(encoding="utf-8")
        .replace("schema_version = 2", "schema_version = 1")
        .replace("VALUES (1, 2, 0, 0", "VALUES (1, 1, 0, 0")
        .replace("PRAGMA user_version = 2", "PRAGMA user_version = 1")
    )
    connection = sqlite3.connect(path)
    connection.executescript(version_1_schema)
    connection.close()

    with pytest.raises(
        UnsupportedIdentitySchemaError,
        match="Unsupported identity database schema 1",
    ):
        IdentityCatalog(path)

    connection = sqlite3.connect(path)
    try:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 1
        assert (
            connection.execute(
                "SELECT schema_version FROM control WHERE singleton = 1"
            ).fetchone()[0]
            == 1
        )
    finally:
        connection.close()


def test_runtime_configuration_and_evidence_are_atomic_and_blob_backed(
    tmp_path: Path,
):
    with IdentityCatalog(tmp_path / "catalog.sqlite") as catalog:
        control = configure(catalog)
        assert control.encoder_key == "miewid-dual-crop-v1"
        assert control.embedding_dimension == 2

        stored = catalog.record_evidence(
            run_id="run-1",
            source="camera-1",
            track_id=7,
            frame_index=0,
            captured_at=datetime(2026, 7, 24, tzinfo=timezone.utc),
            preview_jpeg=b"\xff\xd8preview\xff\xd9",
            embedding=normalized(1.0, 2.0),
            quality=0.8,
            observation_count=1,
        )

        evidence = catalog.connection.execute(
            """
            SELECT typeof(preview_jpeg) AS preview_type,
                   typeof(embedding) AS embedding_type,
                   length(embedding) AS embedding_bytes
            FROM evidence_frames
            WHERE evidence_id = ?
            """,
            (stored.evidence_id,),
        ).fetchone()
        assert evidence["preview_type"] == "blob"
        assert evidence["embedding_type"] == "blob"
        assert evidence["embedding_bytes"] == 8
        assert (
            catalog.connection.execute("SELECT COUNT(*) FROM audit_events").fetchone()[
                0
            ]
            == 2
        )
        assert catalog.control().runtime_revision == 2


def test_audit_events_are_database_immutable(tmp_path: Path):
    with IdentityCatalog(tmp_path / "catalog.sqlite") as catalog:
        configure(catalog)

        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            catalog.connection.execute("UPDATE audit_events SET reason = 'changed'")
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            catalog.connection.execute("DELETE FROM audit_events")


def test_controlled_visit_is_eligible_only_after_explicit_finalization(
    tmp_path: Path,
) -> None:
    with IdentityCatalog(tmp_path / "catalog.sqlite") as catalog:
        configure(catalog)
        stored = None
        for frame_index in range(2):
            stored = catalog.record_evidence(
                run_id="visit-1",
                source="camera-1",
                track_id=7,
                frame_index=frame_index,
                captured_at=datetime(2026, 7, 24, frame_index, tzinfo=timezone.utc),
                preview_jpeg=b"\xff\xd8preview\xff\xd9",
                embedding=normalized(1.0, 2.0),
                quality=0.8,
                observation_count=frame_index + 1,
                evidence_status="insufficient",
            )
        assert stored is not None
        assert (
            catalog.connection.execute(
                "SELECT evidence_status FROM tracklets WHERE tracklet_id = ?",
                (stored.tracklet_id,),
            ).fetchone()[0]
            == "insufficient"
        )

        catalog.finalize_tracklet(
            stored.tracklet_id,
            evidence_status="eligible",
            minimum_evidence_frames=2,
            reason="controlled zone visit cleared cleanly",
        )

        assert (
            catalog.connection.execute(
                "SELECT evidence_status FROM tracklets WHERE tracklet_id = ?",
                (stored.tracklet_id,),
            ).fetchone()[0]
            == "eligible"
        )
        event = catalog.connection.execute(
            """
            SELECT event_type, after_json
            FROM audit_events
            WHERE entity_id = ? AND event_type = 'tracklet_finalized'
            """,
            (stored.tracklet_id,),
        ).fetchone()
        assert event["event_type"] == "tracklet_finalized"
        assert '"evidence_frame_count":2' in event["after_json"]


def test_short_or_switch_risk_tracklet_cannot_be_promoted(tmp_path: Path) -> None:
    with IdentityCatalog(tmp_path / "catalog.sqlite") as catalog:
        configure(catalog)
        short = catalog.record_evidence(
            run_id="visit-short",
            source="camera-1",
            track_id=7,
            frame_index=0,
            captured_at=datetime(2026, 7, 24, tzinfo=timezone.utc),
            preview_jpeg=b"\xff\xd8preview\xff\xd9",
            embedding=normalized(1.0, 2.0),
            quality=0.8,
            observation_count=1,
            evidence_status="insufficient",
        )
        with pytest.raises(IdentityCatalogError, match="enough evidence"):
            catalog.finalize_tracklet(
                short.tracklet_id,
                evidence_status="eligible",
                minimum_evidence_frames=2,
                reason="controlled zone visit cleared cleanly",
            )

        catalog.mark_tracklet_switch_risk(
            short.tracklet_id,
            reason="tracker changed inside controlled zone",
        )
        with pytest.raises(IdentityCatalogError, match="Switch-risk"):
            catalog.finalize_tracklet(
                short.tracklet_id,
                evidence_status="eligible",
                minimum_evidence_frames=1,
                reason="controlled zone visit cleared cleanly",
            )


def test_corrupt_gallery_blob_is_rejected_fail_closed(tmp_path: Path):
    with IdentityCatalog(tmp_path / "catalog.sqlite") as catalog:
        configure(catalog)
        control = catalog.control()
        gallery = catalog.rebuild_gallery(
            expected_operator_revision=control.operator_revision,
            encoder_key="miewid-dual-crop-v1",
            configuration_sha256="a" * 64,
            embedding_dimension=2,
            gallery_frames=4,
        )
        assert gallery.identities == ()

        catalog.connection.execute("PRAGMA ignore_check_constraints = ON")
        catalog.connection.execute(
            """
            INSERT INTO official_identities (
                official_id, status, notes, created_at, updated_at
            ) VALUES ('1', 'active', '', 'now', 'now')
            """
        )
        catalog.connection.execute(
            """
            INSERT INTO visual_identities (
                visual_identity_id, status, created_at, updated_at
            ) VALUES ('vid_corrupt', 'active', 'now', 'now')
            """
        )
        catalog.connection.execute(
            """
            INSERT INTO gallery_items (
                gallery_version, visual_identity_id, official_id, prototype,
                embedding_dimension, evidence_ids_json
            ) VALUES (?, 'vid_corrupt', '1', x'00', 2, '[]')
            """,
            (gallery.gallery_version,),
        )
        catalog.connection.execute("PRAGMA ignore_check_constraints = OFF")

        with pytest.raises(CorruptIdentityBlobError):
            catalog.load_active_gallery(
                expected_operator_revision=control.operator_revision,
                encoder_key="miewid-dual-crop-v1",
                configuration_sha256="a" * 64,
                embedding_dimension=2,
            )
