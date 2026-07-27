from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from aidetector.domain.detections import DetectedObject, Observation
from aidetector.domain.frames import Frame
from aidetector.reid.controlled_zone import ControlledZonePolicy
from aidetector.reid.identity_catalog import IdentityCatalog
from aidetector.reid.models import MIEWID_EMBEDDING_DIMENSION
from aidetector.reid.stage import (
    CandidateFilterPolicy,
    IdentityPolicy,
    IdentityStage,
)


def unit_vector(index: int) -> np.ndarray:
    vector = np.zeros(MIEWID_EMBEDDING_DIMENSION, dtype=np.float32)
    vector[index] = 1.0
    return vector


class FakeEncoder:
    feature_dim = MIEWID_EMBEDDING_DIMENSION

    def __init__(self, vector: np.ndarray | None = None):
        self.vector = vector if vector is not None else unit_vector(0)
        self.calls: list[list[np.ndarray]] = []

    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        self.calls.append([crop.copy() for crop in crops])
        return np.stack([self.vector.copy() for _crop in crops])


def policy(*, max_age: int = 10) -> IdentityPolicy:
    return IdentityPolicy(
        target_label="cow",
        candidate_filter=CandidateFilterPolicy(
            min_area_ratio=0.005,
            max_area_ratio=0.3,
            frame_edge_margin=0.2,
        ),
        controlled_zone=ControlledZonePolicy(
            zone_id="identity_observation",
            x1=0.2,
            y1=0.2,
            x2=0.8,
            y2=0.8,
            minimum_box_inside_ratio=0.9,
            minimum_stable_frames=2,
            clear_frames=2,
        ),
        encoder="miewid-dual-crop-v1",
        similarity_threshold=0.75,
        similarity_margin=0.05,
        query_frames=2,
        gallery_frames=4,
        track_max_age=max_age,
    )


def make_stage(
    tmp_path: Path,
    *,
    encoder: FakeEncoder | None = None,
    max_age: int = 10,
) -> tuple[IdentityStage, IdentityCatalog, FakeEncoder]:
    catalog = IdentityCatalog(tmp_path / "identities.sqlite")
    fake = encoder or FakeEncoder()
    stage = IdentityStage(
        policy=policy(max_age=max_age),
        encoder=fake,  # type: ignore[arg-type]
        catalog=catalog,
        process_run_id="test-run",
    )
    stage.start()
    return stage, catalog, fake


def observation(
    *objects: DetectedObject,
    color: tuple[int, int, int] = (10, 20, 30),
    captured_at: datetime | None = None,
) -> Observation:
    image = np.full((100, 100, 3), color, dtype=np.uint8)
    return Observation(
        Frame(captured_at or datetime(2026, 7, 24, tzinfo=timezone.utc), image),
        tuple(objects),
        {"cow": 0.9},
    )


def cow(track_id: int | None = 7, *, box=(20, 20, 60, 60)) -> DetectedObject:
    return DetectedObject(
        *box,
        label="cow",
        confidence=0.9,
        track_id=track_id,
    )


def seed_confirmed_identity(
    catalog: IdentityCatalog,
    *,
    official_id: str,
    embedding: np.ndarray,
    seed: int,
) -> str:
    stored = []
    for offset, track_id in enumerate((seed, seed + 1)):
        for frame_index in range(2):
            stored.append(
                catalog.record_evidence(
                    run_id=f"gallery-{seed}-{offset}",
                    source=f"gallery-camera-{seed}",
                    track_id=track_id,
                    frame_index=frame_index,
                    captured_at=datetime(
                        2026,
                        7,
                        20 + offset,
                        frame_index,
                        tzinfo=timezone.utc,
                    ),
                    preview_jpeg=b"\xff\xd8gallery\xff\xd9",
                    embedding=embedding,
                    quality=1.0,
                    observation_count=frame_index + 1,
                )
            )
    first_tracklet = stored[0].tracklet_id
    second_tracklet = stored[2].tracklet_id
    visual_identity_id = stored[0].visual_identity_id
    second_visual_identity_id = stored[2].visual_identity_id
    now = "2026-07-24T00:00:00Z"
    with catalog.transaction(immediate=True):
        catalog.connection.execute(
            """
            UPDATE visual_identity_tracklets
            SET visual_identity_id = ?, assignment_kind = 'human_merge'
            WHERE tracklet_id = ?
            """,
            (visual_identity_id, second_tracklet),
        )
        catalog.connection.execute(
            """
            UPDATE visual_identities
            SET status = 'active', updated_at = ?
            WHERE visual_identity_id = ?
            """,
            (now, visual_identity_id),
        )
        catalog.connection.execute(
            """
            UPDATE visual_identities
            SET status = 'merged', merged_into_visual_identity_id = ?, updated_at = ?
            WHERE visual_identity_id = ?
            """,
            (visual_identity_id, now, second_visual_identity_id),
        )
        catalog.connection.execute(
            """
            INSERT INTO official_identities (
                official_id, display_name, status, notes, created_at, updated_at
            ) VALUES (?, NULL, 'active', '', ?, ?)
            """,
            (official_id, now, now),
        )
        catalog.connection.execute(
            """
            INSERT INTO mappings (
                mapping_id, visual_identity_id, official_id, state,
                provisional_tracklet_id, confirmation_tracklet_id, version,
                created_at, updated_at
            ) VALUES (?, ?, ?, 'confirmed', ?, ?, 1, ?, ?)
            """,
            (
                f"map_{seed}",
                visual_identity_id,
                official_id,
                first_tracklet,
                second_tracklet,
                now,
                now,
            ),
        )
        catalog.connection.execute(
            """
            UPDATE control
            SET operator_revision = operator_revision + 1, updated_at = ?
            WHERE singleton = 1
            """,
            (now,),
        )
    return visual_identity_id


def test_stage_uses_raw_rgb_primary_boxes_and_two_frame_consensus(tmp_path: Path):
    stage, catalog, encoder = make_stage(tmp_path)
    horse = DetectedObject(20, 20, 60, 60, "horse", 0.9, 7)

    first = stage.enrich("camera", observation(cow(), horse))
    second = stage.enrich("camera", observation(cow(), horse))
    third = stage.enrich("camera", observation(cow(), horse))
    assert (
        catalog.connection.execute("SELECT evidence_status FROM tracklets").fetchone()[
            0
        ]
        == "insufficient"
    )
    stage.enrich("camera", observation())
    stage.enrich("camera", observation())

    assert len(encoder.calls) == 2
    assert [len(call) for call in encoder.calls] == [1, 1]
    assert np.all(encoder.calls[0][0] == np.asarray((30, 20, 10), dtype=np.uint8))
    assert first.objects[0].identity is not None
    assert first.objects[0].identity.status == "insufficient_evidence"
    assert second.objects[0].identity is not None
    assert second.objects[0].identity.status == "insufficient_evidence"
    assert third.objects[0].identity is not None
    assert third.objects[0].identity.status == "unknown"
    assert first.objects[0].identity.visual_identity_id is None
    assert second.objects[0].identity.visual_identity_id == (
        third.objects[0].identity.visual_identity_id
    )
    assert all(item.identity is None for item in (first.objects[1], second.objects[1]))

    row = catalog.connection.execute(
        """
        SELECT observation_count, evidence_status,
               typeof(preview_jpeg) AS preview_type
        FROM tracklets
        """
    ).fetchone()
    assert dict(row) == {
        "observation_count": 2,
        "evidence_status": "eligible",
        "preview_type": "blob",
    }
    assert (
        catalog.connection.execute("SELECT COUNT(*) FROM evidence_frames").fetchone()[0]
        == 2
    )
    stage.close()


def test_stage_taints_a_track_id_replaced_before_zone_clearance(tmp_path: Path):
    stage, catalog, encoder = make_stage(tmp_path)
    stage.enrich("camera", observation(cow(7)))
    stage.enrich("camera", observation(cow(7)))

    replaced = stage.enrich("camera", observation(cow(8)))

    assert replaced.objects[0].identity is not None
    assert replaced.objects[0].identity.status == "switch_risk"
    assert len(encoder.calls) == 1
    row = catalog.connection.execute(
        """
        SELECT evidence_status
        FROM tracklets
        WHERE source = 'camera' AND track_id = 7
        """
    ).fetchone()
    assert row["evidence_status"] == "switch_risk"
    stage.close()


def test_stage_suppresses_cached_identity_outside_the_zone(tmp_path: Path):
    stage, _catalog, encoder = make_stage(tmp_path)
    stage.enrich("camera", observation(cow(7)))
    stage.enrich("camera", observation(cow(7)))
    inside = stage.enrich("camera", observation(cow(7)))
    calls_before_exit = len(encoder.calls)

    outside = stage.enrich(
        "camera",
        observation(cow(7, box=(0, 0, 10, 10))),
    )

    assert inside.objects[0].identity is not None
    assert inside.objects[0].identity.visual_identity_id is not None
    assert outside.objects[0].identity is not None
    assert outside.objects[0].identity.status == "insufficient_evidence"
    assert outside.objects[0].identity.visual_identity_id is None
    assert len(encoder.calls) == calls_before_exit
    stage.close()


def test_stage_shutdown_never_promotes_an_active_zone_visit(tmp_path: Path):
    database = tmp_path / "identities.sqlite"
    stage, catalog, _encoder = make_stage(tmp_path)
    stage.enrich("camera", observation(cow(7)))
    stage.enrich("camera", observation(cow(7)))
    stage.enrich("camera", observation(cow(7)))

    assert (
        catalog.connection.execute("SELECT evidence_status FROM tracklets").fetchone()[
            0
        ]
        == "insufficient"
    )
    stage.close()

    connection = sqlite3.connect(database)
    try:
        assert (
            connection.execute("SELECT evidence_status FROM tracklets").fetchone()[0]
            == "insufficient"
        )
        assert (
            connection.execute(
                """
                SELECT COUNT(*)
                FROM audit_events
                WHERE event_type = 'tracklet_finalized'
                  AND reason = 'detector stopped before controlled-zone clearance'
                """
            ).fetchone()[0]
            == 1
        )
    finally:
        connection.close()


def test_stage_scopes_track_ids_and_zone_visits_by_source(tmp_path: Path):
    stage, _catalog, encoder = make_stage(tmp_path)
    stage.enrich("camera-a", observation(cow(7)))
    first_a = stage.enrich("camera-a", observation(cow(7)))
    stage.enrich("camera-b", observation(cow(7)))
    first_b = stage.enrich("camera-b", observation(cow(7)))

    assert [len(call) for call in encoder.calls] == [1, 1]
    identity_a7 = first_a.objects[0].identity
    identity_b7 = first_b.objects[0].identity
    assert identity_a7 is not None
    assert identity_b7 is not None
    assert identity_a7.visual_identity_id != identity_b7.visual_identity_id
    stage.close()


def test_stage_matches_gallery_but_never_assigns_prediction_as_truth(tmp_path: Path):
    stage, catalog, _encoder = make_stage(tmp_path)
    gallery_visual_id = seed_confirmed_identity(
        catalog,
        official_id="NL-123",
        embedding=unit_vector(0),
        seed=100,
    )

    first = stage.enrich("camera", observation(cow(7)))
    second = stage.enrich("camera", observation(cow(7)))
    third = stage.enrich("camera", observation(cow(7)))

    assert first.objects[0].identity is not None
    assert first.objects[0].identity.status == "insufficient_evidence"
    assert second.objects[0].identity is not None
    assert second.objects[0].identity.status == "insufficient_evidence"
    result = third.objects[0].identity
    assert result is not None
    assert result.status == "matched"
    assert result.visual_identity_id == gallery_visual_id
    assert result.official_id == "NL-123"
    assert result.similarity == 1.0
    assert result.margin == 2.0
    assert result.gallery_version is not None

    runtime_assignment = catalog.connection.execute(
        """
        SELECT vit.visual_identity_id, t.predicted_visual_identity_id
        FROM tracklets t
        JOIN visual_identity_tracklets vit USING (tracklet_id)
        WHERE t.source = 'camera'
        """
    ).fetchone()
    assert runtime_assignment["visual_identity_id"] != gallery_visual_id
    assert runtime_assignment["predicted_visual_identity_id"] == gallery_visual_id
    assert (
        catalog.connection.execute(
            """
        SELECT COUNT(*)
        FROM mappings
        WHERE visual_identity_id = ?
        """,
            (runtime_assignment["visual_identity_id"],),
        ).fetchone()[0]
        == 0
    )
    stage.close()


def test_similarity_and_runner_up_gates_abstain_without_official_id(tmp_path: Path):
    stage, catalog, _encoder = make_stage(tmp_path)
    seed_confirmed_identity(
        catalog,
        official_id="A",
        embedding=unit_vector(0),
        seed=100,
    )
    runner_up = unit_vector(0)
    runner_up[0] = np.float32(0.99)
    runner_up[1] = np.float32(np.sqrt(1.0 - 0.99**2))
    seed_confirmed_identity(
        catalog,
        official_id="B",
        embedding=runner_up,
        seed=200,
    )

    stage.enrich("camera", observation(cow(7)))
    stage.enrich("camera", observation(cow(7)))
    ambiguous = stage.enrich("camera", observation(cow(7)))

    result = ambiguous.objects[0].identity
    assert result is not None
    assert result.status == "ambiguous"
    assert result.official_id is None
    assert result.similarity == pytest.approx(1.0)
    assert result.margin == pytest.approx(0.01, abs=1e-6)
    stage.close()


def test_conflicting_confirmed_matches_mark_the_track_as_switch_risk(tmp_path: Path):
    encoder = FakeEncoder(unit_vector(0))
    stage, catalog, _encoder = make_stage(tmp_path, encoder=encoder)
    seed_confirmed_identity(
        catalog,
        official_id="A",
        embedding=unit_vector(0),
        seed=100,
    )
    seed_confirmed_identity(
        catalog,
        official_id="B",
        embedding=unit_vector(1),
        seed=200,
    )

    stage.enrich("camera", observation(cow(7)))
    stage.enrich("camera", observation(cow(7)))
    matched = stage.enrich("camera", observation(cow(7)))
    assert matched.objects[0].identity is not None
    assert matched.objects[0].identity.official_id == "A"

    encoder.vector = unit_vector(1)
    stage.enrich("camera", observation(cow(7)))
    switched = stage.enrich("camera", observation(cow(7)))

    assert switched.objects[0].identity is not None
    assert switched.objects[0].identity.status == "switch_risk"
    assert switched.objects[0].identity.official_id is None
    assert (
        catalog.connection.execute(
            """
        SELECT evidence_status FROM tracklets WHERE source = 'camera'
        """
        ).fetchone()[0]
        == "switch_risk"
    )
    stage.close()


def test_duplicate_track_ids_and_filtered_boxes_never_enter_encoder(tmp_path: Path):
    stage, _catalog, encoder = make_stage(tmp_path)

    duplicate = stage.enrich("camera", observation(cow(7), cow(7)))
    edge = stage.enrich(
        "camera",
        observation(cow(8, box=(0, 0, 10, 10))),
    )

    assert all(
        item.identity is not None and item.identity.status == "switch_risk"
        for item in duplicate.objects
    )
    assert edge.objects[0].identity is not None
    assert edge.objects[0].identity.status == "insufficient_evidence"
    assert encoder.calls == []
    stage.close()


def test_duplicate_existing_track_id_persists_switch_risk(tmp_path: Path):
    stage, catalog, encoder = make_stage(tmp_path)
    stage.enrich("camera", observation(cow(7)))
    stage.enrich("camera", observation(cow(7)))
    calls_before_duplicate = len(encoder.calls)

    duplicate = stage.enrich("camera", observation(cow(7), cow(7)))

    assert all(
        item.identity is not None and item.identity.status == "switch_risk"
        for item in duplicate.objects
    )
    assert len(encoder.calls) == calls_before_duplicate
    row = catalog.connection.execute(
        """
        SELECT evidence_status
        FROM tracklets
        WHERE source = 'camera' AND track_id = 7
        """
    ).fetchone()
    assert row["evidence_status"] == "switch_risk"
    assert (
        catalog.connection.execute(
            """
        SELECT COUNT(*)
        FROM audit_events
        WHERE event_type = 'tracklet_switch_risk'
        """
        ).fetchone()[0]
        == 1
    )
    stage.close()


def test_track_id_reuse_after_max_age_creates_a_new_tracklet(tmp_path: Path):
    stage, catalog, _encoder = make_stage(tmp_path, max_age=2)

    stage.enrich("camera", observation(cow(7)))
    first = stage.enrich("camera", observation(cow(7)))
    stage.enrich("camera", observation())
    stage.enrich("camera", observation())
    stage.enrich("camera", observation(cow(7)))
    reused = stage.enrich("camera", observation(cow(7)))

    assert first.objects[0].identity is not None
    assert reused.objects[0].identity is not None
    assert first.objects[0].identity.visual_identity_id != (
        reused.objects[0].identity.visual_identity_id
    )
    assert (
        catalog.connection.execute(
            "SELECT COUNT(*) FROM tracklets WHERE source = 'camera' AND track_id = 7"
        ).fetchone()[0]
        == 2
    )
    stage.close()


def test_corrupt_evidence_disables_output_until_operator_revision_changes(
    tmp_path: Path,
):
    stage, catalog, encoder = make_stage(tmp_path)
    seed_confirmed_identity(
        catalog,
        official_id="NL-123",
        embedding=unit_vector(0),
        seed=100,
    )
    stage.enrich("camera", observation(cow(7)))
    stage.enrich("camera", observation(cow(7)))
    stage.enrich("camera", observation(cow(7)))

    catalog.connection.execute("PRAGMA ignore_check_constraints = ON")
    catalog.connection.execute(
        """
        UPDATE evidence_frames
        SET embedding = x'00'
        WHERE evidence_id = (
            SELECT json_each.value
            FROM gallery_items
            JOIN control
              ON gallery_items.gallery_version = control.active_gallery_version
            JOIN json_each(gallery_items.evidence_ids_json)
            WHERE control.singleton = 1
            LIMIT 1
        )
        """
    )
    catalog.connection.execute("PRAGMA ignore_check_constraints = OFF")
    catalog.connection.execute(
        """
        UPDATE control
        SET operator_revision = operator_revision + 1
        WHERE singleton = 1
        """
    )
    calls_before_error = len(encoder.calls)

    failed = stage.enrich("camera", observation(cow(8)))
    repeated = stage.enrich("camera", observation(cow(8)))

    assert failed.objects[0].identity is not None
    assert failed.objects[0].identity.status == "error"
    assert repeated.objects[0].identity is not None
    assert repeated.objects[0].identity.status == "error"
    assert len(encoder.calls) == calls_before_error
    control = catalog.control()
    assert control.active_gallery_version is None
    assert "BLOB" in (control.last_identity_error or "")
    with np.testing.assert_raises(sqlite3.IntegrityError):
        catalog.connection.execute("DELETE FROM audit_events")
    stage.close()
