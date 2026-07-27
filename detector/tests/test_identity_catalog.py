import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from aidetector.reid.identity_catalog import (
    SCHEMA_VERSION,
    IdentityCatalog,
    IdentityCatalogError,
    UnsupportedIdentitySchemaError,
)


def configure(catalog: IdentityCatalog) -> None:
    catalog.configure_runtime(
        encoder_key="miewid-dual-crop-v1",
        embedding_dimension=2,
        configuration_sha256="a" * 64,
    )


def record(catalog: IdentityCatalog, frame_index: int = 0):
    return catalog.record_evidence(
        run_id="visit-1",
        source="camera-1",
        track_id=7,
        frame_index=frame_index,
        captured_at=datetime(2026, 7, 24, frame_index, tzinfo=timezone.utc),
        preview_jpeg=b"\xff\xd8preview\xff\xd9",
        embedding=np.asarray((1.0, 0.0), dtype=np.float32),
        observation_count=frame_index + 1,
        evidence_status="insufficient",
    )


def test_catalog_initializes_once_and_rejects_legacy_schema(tmp_path: Path):
    path = tmp_path / "identities" / "catalog.sqlite"
    with IdentityCatalog(path) as catalog:
        assert catalog.connection.execute("PRAGMA user_version").fetchone()[0] == (
            SCHEMA_VERSION
        )
        assert catalog.connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert catalog.connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        configure(catalog)
        assert catalog.gallery().identities == ()

    legacy = tmp_path / "legacy.sqlite"
    connection = sqlite3.connect(legacy)
    connection.execute("CREATE TABLE identities (identity TEXT PRIMARY KEY)")
    connection.close()
    with pytest.raises(UnsupportedIdentitySchemaError):
        IdentityCatalog(legacy)


def test_evidence_is_blob_backed_and_becomes_eligible_after_two_frames(
    tmp_path: Path,
):
    with IdentityCatalog(tmp_path / "catalog.sqlite") as catalog:
        configure(catalog)
        stored = record(catalog)
        stored = record(catalog, 1)

        evidence = catalog.connection.execute(
            """
            SELECT typeof(preview_jpeg), typeof(embedding), length(embedding)
            FROM evidence_frames
            WHERE tracklet_id = ? AND frame_index = 1
            """,
            (stored.tracklet_id,),
        ).fetchone()
        assert tuple(evidence) == ("blob", "blob", 8)

        catalog.finalize_tracklet(stored.tracklet_id, evidence_status="eligible")
        status = catalog.connection.execute(
            "SELECT evidence_status FROM tracklets WHERE tracklet_id = ?",
            (stored.tracklet_id,),
        ).fetchone()[0]
        assert status == "eligible"


def test_switch_risk_evidence_cannot_be_promoted(tmp_path: Path):
    with IdentityCatalog(tmp_path / "catalog.sqlite") as catalog:
        configure(catalog)
        stored = record(catalog)
        record(catalog, 1)
        catalog.mark_tracklet_switch_risk(stored.tracklet_id)

        with pytest.raises(IdentityCatalogError, match="Switch-risk"):
            catalog.finalize_tracklet(stored.tracklet_id, evidence_status="eligible")
