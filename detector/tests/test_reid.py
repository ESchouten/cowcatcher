from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np
import pytest

from aidetector.benchmarks.reid.datasets import discover_public_dataset
from aidetector.benchmarks.reid.enrollment import (
    LabeledTrack,
    camera_disjoint_identity_metrics,
    production_open_enrollment_metrics,
)
from aidetector.benchmarks.reid.video import (
    GroundTruth,
    VideoObservation,
    evaluate_observations,
)
from aidetector.reid.catalog import CatalogPolicy, SqliteIdentityCatalog
from aidetector.reid.enrollment import (
    EnrollmentTrack,
    cluster_known_count,
    cluster_tracklets,
    cluster_tracks,
    finalize_enrollment,
    finalize_pending_enrollment,
)
from aidetector.reid.gallery import IdentityGallery
from aidetector.reid.miewid import MiewIdEncoder, _preprocess
from aidetector.reid.policy import DEFAULT_REID_POLICY
from aidetector.reid.segmentation import (
    LocalizerSettings,
    box_allowed,
    candidates_from_result,
    masked_candidate,
)
from aidetector.reid.store import TrackletStore
from aidetector.domain.detections import DetectedObject as Crop
from aidetector.domain.frames import Frame
from aidetector.domain.identity import (
    IdentityCandidate,
    IdentityMatch,
    IdentityResult,
    TrackletSnapshot,
)
from aidetector.pipeline.identity_provider import (
    IdentityBatch,
    SamplingDecision,
    TrackedIdentityProvider,
)
from aidetector.pipeline.identity_tracking import TrackIdentityAggregator


class Scalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class Box:
    def __init__(
        self,
        xyxy,
        *,
        confidence: float = 0.9,
        track_id: int | None = 1,
        class_id: int = 19,
    ):
        self.xyxy = [xyxy]
        self.conf = Scalar(confidence)
        self.id = Scalar(track_id) if track_id is not None else None
        self.cls = Scalar(class_id)


class MaskData:
    def __init__(self, value):
        self.value = value

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class Masks:
    def __init__(self, values):
        self.data = [MaskData(value) for value in values]


class Result:
    def __init__(self, boxes, masks, names=None):
        self.boxes = boxes
        self.masks = Masks(masks)
        self.names = names or {19: "cow"}


def test_miewid_preprocessing_uses_rgb_imagenet_input():
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    image[:] = [0, 0, 255]

    result = _preprocess(image)

    assert result.shape == (1, 3, 440, 440)
    assert result.dtype == np.float32
    assert result.flags.c_contiguous
    assert result[0, :, 0, 0] == pytest.approx(
        np.asarray([(1 - 0.485) / 0.229, -0.456 / 0.224, -0.406 / 0.225])
    )


def test_miewid_encoder_normalizes_onnx_embeddings():
    inputs = []

    class Session:
        def get_inputs(self):
            return [type("Input", (), {"name": "input", "shape": [1, 3, 440, 440]})()]

        def get_outputs(self):
            return [type("Output", (), {"name": "output", "shape": [1, 2]})()]

        def run(self, outputs, values):
            inputs.append((outputs, values))
            return [np.asarray([[3, 4]], dtype=np.float32)]

    encoder = MiewIdEncoder(Session())

    result = encoder.embed([np.zeros((10, 20, 3), dtype=np.uint8)])

    assert result[0] == pytest.approx([0.6, 0.8])
    assert inputs[0][0] == ["output"]
    assert inputs[0][1]["input"].shape == (1, 3, 440, 440)
    assert encoder.embed([]).shape == (0, 2)


def test_box_allowed_applies_area_and_margin_filters():
    assert box_allowed((10, 10, 50, 50), (100, 100), 0.1, 0.3)
    assert not box_allowed((70, 70, 75, 75), (100, 100), 0.1, 0.3)
    assert not box_allowed((0, 20, 10, 40), (100, 100), 0, 1, margin=0.1)


def test_masked_candidate_replaces_background_and_crops_mask():
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    frame[:] = [10, 20, 30]
    frame[1:3, 1:3] = [100, 110, 120]
    mask = np.zeros((4, 4), dtype=bool)
    mask[1:3, 1:3] = True

    candidate = masked_candidate(frame, mask, "cow", 0.9)

    assert candidate is not None
    assert candidate.detection == Crop(1, 1, 3, 3, label="cow", confidence=0.9)
    assert np.all(candidate.image == [100, 110, 120])


def test_candidates_from_result_keeps_cows_and_track_ids_only():
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    cow_mask = np.zeros((20, 20), dtype=bool)
    cow_mask[2:10, 2:10] = True
    horse_mask = np.zeros((20, 20), dtype=bool)
    horse_mask[10:18, 10:18] = True
    result = Result(
        [Box([2, 2, 10, 10], track_id=7), Box([10, 10, 18, 18], class_id=17)],
        [cow_mask, horse_mask],
        {19: "cow", 17: "horse"},
    )
    settings = LocalizerSettings("cow", 0.1, 0, 1, 0)

    candidates = candidates_from_result(result, frame, settings)

    assert len(candidates) == 1
    assert candidates[0].detection.track_id == 7
    assert candidates[0].detection.label == "cow"


def test_candidates_from_result_applies_identity_confidence():
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    mask = np.ones((20, 20), dtype=bool)
    result = Result([Box([0, 0, 20, 20], confidence=0.4)], [mask])

    candidates = candidates_from_result(
        result,
        frame,
        LocalizerSettings("cow", 0.5, 0, 1, 0),
    )

    assert candidates == []


def test_gallery_matches_best_sample_per_identity():
    gallery = IdentityGallery(
        np.array([[1, 0], [0.9, 0.1], [0, 1]], dtype=np.float32),
        ["001", "001", "002"],
        ["NL-123", "NL-123", "NL-456"],
        match_threshold=0.8,
        match_margin=0.05,
    )

    result = gallery.match(np.array([1, 0], dtype=np.float32))

    assert result is not None
    assert result == IdentityMatch("001", "NL-123", 1, 1)
    assert result.result == IdentityResult("NL-123", 1)


def test_gallery_is_not_biased_toward_identities_with_more_samples():
    gallery = IdentityGallery(
        np.array(
            [[0.8, 0.6], [0.8, -0.6], [0.9, np.sqrt(0.19)]],
            dtype=np.float32,
        ),
        ["many", "many", "best"],
        ["many", "many", "best"],
        match_threshold=0,
        match_margin=0,
    )

    result = gallery.match(np.array([1, 0], dtype=np.float32))

    assert result is not None
    assert result.key == "best"
    assert result.similarity == pytest.approx(0.9)
    assert result.margin == pytest.approx(0.1)


def test_gallery_rejects_low_or_ambiguous_matches():
    gallery = IdentityGallery(
        np.array([[1, 0], [0.99, 0.01]], dtype=np.float32),
        ["001", "002"],
        ["001", "002"],
        match_threshold=0,
        match_margin=0.1,
    )

    assert gallery.match(np.array([1, 0], dtype=np.float32)) is None
    with pytest.raises(ValueError, match="dimension does not match"):
        gallery.match(np.array([1, 0, 0], dtype=np.float32))


def test_camera_disjoint_identity_metrics_include_unknown_false_accepts():
    tracks = [
        LabeledTrack(EnrollmentTrack("a-1", np.asarray([1, 0])), "a", "1"),
        LabeledTrack(EnrollmentTrack("a-2", np.asarray([1, 0])), "a", "2"),
        LabeledTrack(EnrollmentTrack("b-1", np.asarray([0, 1])), "b", "1"),
        LabeledTrack(EnrollmentTrack("b-2", np.asarray([0, 1])), "b", "2"),
    ]
    unknown = [
        LabeledTrack(EnrollmentTrack("c-1", np.asarray([1, 0])), "c", "1"),
        LabeledTrack(EnrollmentTrack("c-2", np.asarray([0, 1])), "c", "2"),
    ]

    metrics = camera_disjoint_identity_metrics(
        tracks,
        unknown_tracks=unknown,
    )

    assert metrics["top1_accuracy"] == 1
    assert metrics["identification_rate"] == 1
    assert metrics["misidentification_rate"] == 0
    assert metrics["unknown_false_acceptance_rate"] == 1


def test_production_open_enrollment_metrics_use_runtime_defaults():
    tracks = [
        LabeledTrack(EnrollmentTrack("a-1", np.asarray([1, 0])), "a", "1"),
        LabeledTrack(EnrollmentTrack("a-2", np.asarray([0.99, 0.01])), "a", "2"),
        LabeledTrack(EnrollmentTrack("b-1", np.asarray([0, 1])), "b", "1"),
        LabeledTrack(EnrollmentTrack("b-2", np.asarray([0.01, 0.99])), "b", "2"),
    ]

    metrics = production_open_enrollment_metrics(tracks)

    assert metrics["clusters"] == 2
    assert metrics["complete_identity_rate"] == 1
    assert metrics["merge_errors"] == 0


def test_track_identity_aggregates_and_caches_embeddings():
    calls = []

    class Gallery:
        def match(self, embedding):
            calls.append(embedding)
            return IdentityMatch("identity-001", "NL-123", 0.9, 0.2)

    tracks = TrackIdentityAggregator(Gallery(), samples=3, max_age=10)
    candidates = []
    for embedding in ([1, 0], [0.8, 0.2], [1, 0], [0, 1]):
        candidate = IdentityCandidate(
            Crop(0, 0, 10, 10, label="cow", track_id=1),
            np.zeros((10, 10, 3), dtype=np.uint8),
        )
        update = tracks.apply(
            "camera",
            [candidate],
            np.asarray([embedding], dtype=np.float32),
        )
        candidates.append(update.candidates[0])

    assert candidates[0].detection.identities == ()
    assert candidates[1].detection.identities == ()
    assert candidates[2].detection.identities == (IdentityResult("NL-123", 0.9),)
    assert candidates[3].detection.identities == (IdentityResult("NL-123", 0.9),)
    assert len(calls) == 1


def test_track_identity_rechecks_each_complete_sample_window():
    matches = iter(
        (
            IdentityMatch("identity-001", "NL-123", 0.9, 0.2),
            None,
        )
    )
    gallery = type("Gallery", (), {"match": lambda _self, _embedding: next(matches)})()
    tracks = TrackIdentityAggregator(gallery, samples=2, max_age=10)
    candidates = []
    for _ in range(4):
        candidate = IdentityCandidate(
            Crop(0, 0, 10, 10, label="cow", track_id=1),
            np.zeros((10, 10, 3), dtype=np.uint8),
        )
        update = tracks.apply(
            "camera",
            [candidate],
            np.asarray([[1, 0]], dtype=np.float32),
        )
        candidates.append(update.candidates[0])

    assert candidates[1].detection.identities == (IdentityResult("NL-123", 0.9),)
    assert candidates[2].detection.identities == (IdentityResult("NL-123", 0.9),)
    assert candidates[3].detection.identities == ()


def test_track_identity_rechecks_active_tracks_after_gallery_change():
    first = type(
        "Gallery",
        (),
        {"match": lambda _self, _embedding: IdentityMatch("identity-1", "001", 0.9, 1)},
    )()
    second = type(
        "Gallery",
        (),
        {
            "match": lambda _self, _embedding: IdentityMatch(
                "identity-1", "NL-123", 0.9, 1
            )
        },
    )()
    tracks = TrackIdentityAggregator(first, samples=1, max_age=10)
    candidate = IdentityCandidate(Crop(0, 0, 10, 10, track_id=1), np.zeros((10, 10, 3)))

    candidate = tracks.apply("camera", [candidate], np.asarray([[1, 0]])).candidates[0]
    tracks.set_gallery(second)
    candidate = tracks.apply("camera", [candidate], np.asarray([[1, 0]])).candidates[0]

    assert candidate.detection.identities == (IdentityResult("NL-123", 0.9),)


def test_track_identity_keeps_cached_match_until_the_next_sample_window():
    first = type(
        "Gallery",
        (),
        {"match": lambda _self, _embedding: IdentityMatch("identity-1", "001", 0.9, 1)},
    )()
    second = type(
        "Gallery",
        (),
        {
            "match": lambda _self, _embedding: IdentityMatch(
                "identity-1", "NL-123", 0.9, 1
            )
        },
    )()
    tracks = TrackIdentityAggregator(first, samples=2, max_age=10)
    candidate = IdentityCandidate(Crop(0, 0, 10, 10, track_id=1), np.zeros((10, 10, 3)))

    for _ in range(2):
        candidate = tracks.apply(
            "camera", [candidate], np.asarray([[1, 0]])
        ).candidates[0]
    tracks.set_gallery(second)
    candidate = tracks.apply("camera", [candidate], np.asarray([[1, 0]])).candidates[0]

    assert candidate.detection.identities == (IdentityResult("001", 0.9),)


def test_track_identity_returns_cumulative_enrollment_snapshots():
    tracks = TrackIdentityAggregator(None, samples=2, max_age=10)
    snapshots = []
    for embedding in ([1, 0], [1, 0], [0, 1], [0, 1]):
        candidate = IdentityCandidate(
            Crop(0, 0, 10, 10, track_id=1),
            np.zeros((10, 10, 3), dtype=np.uint8),
        )
        snapshots.extend(
            tracks.apply("camera", [candidate], np.asarray([embedding])).snapshots
        )

    assert [snapshot.observations for snapshot in snapshots] == [2, 4]
    assert snapshots[-1].embedding == pytest.approx(np.asarray([1, 1]) / np.sqrt(2))


def test_track_identity_stores_at_most_one_online_sample_per_track():
    gallery = type(
        "Gallery",
        (),
        {
            "match": lambda _self, _embedding: IdentityMatch(
                "identity-1", "001", 0.95, 1
            )
        },
    )()
    tracks = TrackIdentityAggregator(gallery, samples=1, max_age=10)
    candidate = IdentityCandidate(Crop(0, 0, 10, 10, track_id=1), np.zeros((10, 10, 3)))

    first = tracks.apply("camera", [candidate], np.asarray([[1, 0]]))
    tracks.stop_sampling("camera", 1)
    second = tracks.apply("camera", [candidate], np.asarray([[1, 0]]))

    assert len(first.snapshots) == 1
    assert second.snapshots == ()


def test_catalog_learns_only_when_full_track_still_matches_same_identity():
    gallery = IdentityGallery(
        np.asarray([[1, 0], [0, 1]], dtype=np.float32),
        ["identity-1", "identity-2"],
        ["001", "002"],
        match_threshold=0.75,
        match_margin=0.05,
    )
    updated = []
    store = type(
        "Store",
        (),
        {
            "is_finalized": lambda _self: True,
            "update_identity": lambda _self, snapshot, **_kwargs: updated.append(
                snapshot
            ),
            "update_pending": lambda _self, _snapshot: None,
        },
    )()
    catalog = SqliteIdentityCatalog(
        store,
        CatalogPolicy(match_threshold=0.75, match_margin=0, track_samples=5),
    )
    catalog.gallery = gallery
    snapshot = TrackletSnapshot(
        "camera",
        1,
        1,
        10,
        10,
        np.asarray([1, 0]),
        np.zeros((10, 10, 3)),
        "identity-1",
    )

    assert catalog.record(snapshot) is SamplingDecision.STOP
    assert updated == [snapshot]

    conflicting = TrackletSnapshot(
        "camera",
        1,
        1,
        10,
        10,
        np.asarray([0, 1]),
        np.zeros((10, 10, 3)),
        "identity-1",
    )
    assert catalog.record(conflicting) is SamplingDecision.CONTINUE
    assert updated == [snapshot]


def test_catalog_does_not_learn_from_the_first_sample_window():
    gallery = IdentityGallery(
        np.asarray([[1, 0]], dtype=np.float32),
        ["identity-1"],
        ["001"],
        match_threshold=0.75,
        match_margin=0.05,
    )
    updated = []
    store = type(
        "Store",
        (),
        {
            "is_finalized": lambda _self: True,
            "update_identity": lambda _self, snapshot, **_kwargs: updated.append(
                snapshot
            ),
            "update_pending": lambda _self, _snapshot: None,
        },
    )()
    catalog = SqliteIdentityCatalog(
        store,
        CatalogPolicy(match_threshold=0.75, match_margin=0, track_samples=5),
    )
    catalog.gallery = gallery
    snapshot = TrackletSnapshot(
        "camera",
        1,
        1,
        5,
        5,
        np.asarray([1, 0]),
        np.zeros((10, 10, 3)),
        "identity-1",
    )

    assert catalog.record(snapshot) is SamplingDecision.CONTINUE
    assert updated == []


def test_pipeline_tracks_fallback_candidates_and_attaches_identity():
    candidate = IdentityCandidate(
        Crop(1, 2, 8, 9, label="cow", confidence=0.8, track_id=1),
        np.zeros((7, 7, 3), dtype=np.uint8),
    )
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    candidate_source = type(
        "CandidateSource",
        (),
        {
            "batches": lambda _self, batch: [
                IdentityBatch("camera", batch["camera"], (candidate,))
            ]
        },
    )()
    encoder = type(
        "Encoder",
        (),
        {"embed": lambda _self, _images: np.array([[1, 0]], dtype=np.float32)},
    )()
    gallery = type(
        "Gallery",
        (),
        {
            "match": lambda _self, _embedding: IdentityMatch(
                "identity-001", "identity-001", 0.95, 1
            )
        },
    )()
    catalog = type(
        "Catalog",
        (),
        {
            "sync": lambda _self: gallery,
            "record": lambda _self, _snapshot: SamplingDecision.STOP,
            "close": lambda _self: None,
        },
    )()
    pipeline = TrackedIdentityProvider(
        candidates=candidate_source,
        encoder=encoder,
        catalog=catalog,
        tracks=TrackIdentityAggregator(gallery, samples=1, max_age=10),
    )

    result = pipeline.process({"camera": [Frame(datetime(2026, 1, 1), frame)]})

    assert result[0].candidates[0].detection.identities == (
        IdentityResult("identity-001", 0.95),
    )


def test_pipeline_stores_only_unknown_tracks_as_pending():
    pending = []
    gallery = type("Gallery", (), {"match": lambda _self, _embedding: None})()
    encoder = type(
        "Encoder",
        (),
        {"embed": lambda _self, _images: np.asarray([[1, 0]], dtype=np.float32)},
    )()
    catalog = type(
        "Catalog",
        (),
        {
            "sync": lambda _self: gallery,
            "record": lambda _self, snapshot: (
                pending.append(snapshot) or SamplingDecision.CONTINUE
            ),
            "close": lambda _self: None,
        },
    )()
    candidate = IdentityCandidate(
        Crop(0, 0, 10, 10, track_id=1),
        np.zeros((10, 10, 3), dtype=np.uint8),
    )
    candidate_source = type(
        "CandidateSource",
        (),
        {
            "batches": lambda _self, batch: [
                IdentityBatch("camera", batch["camera"], (candidate,))
            ]
        },
    )()
    pipeline = TrackedIdentityProvider(
        candidates=candidate_source,
        encoder=encoder,
        catalog=catalog,
        tracks=TrackIdentityAggregator(gallery, samples=1, max_age=10),
    )

    pipeline.process(
        {"camera": [Frame(datetime(2026, 1, 1), np.zeros((10, 10, 3), dtype=np.uint8))]}
    )

    assert len(pending) == 1
    assert pending[0].identity_key is None


def test_enrollment_clusters_similar_tracks_and_respects_conflicts():
    tracks = [
        EnrollmentTrack("camera-1/1", np.asarray([1, 0])),
        EnrollmentTrack("camera-2/7", np.asarray([0.99, 0.01])),
        EnrollmentTrack(
            "camera-1/2",
            np.asarray([0, 1]),
            cannot_link=frozenset({"camera-2/4"}),
        ),
        EnrollmentTrack("camera-2/4", np.asarray([0.01, 0.99])),
    ]

    assignments = cluster_tracks(tracks, similarity_threshold=0.95, neighbors=2)

    assert assignments["camera-1/1"] == assignments["camera-2/7"]
    assert assignments["camera-1/2"] != assignments["camera-2/4"]


def test_enrollment_known_count_and_ambiguous_clustering():
    tracks = [
        EnrollmentTrack("a", np.asarray([1.0, 0.0])),
        EnrollmentTrack("b", np.asarray([0.99, 0.01])),
        EnrollmentTrack("c", np.asarray([0.98, 0.02])),
    ]

    ambiguous = cluster_tracklets(
        tracks,
        similarity_threshold=0.9,
        margin_threshold=0.01,
    )
    known = cluster_known_count(tracks, 2)

    assert len(set(ambiguous.values())) == 3
    assert len(set(known.values())) == 2


def test_catalog_finalizes_enrollment_and_loads_gallery(tmp_path):
    store = TrackletStore(tmp_path / "cows.sqlite", session="scan")
    catalog = SqliteIdentityCatalog(
        store,
        CatalogPolicy(
            match_threshold=0.75,
            match_margin=0,
            track_samples=1,
            enrollment_identity_count=1,
        ),
    )
    assert catalog.initialize("test-model", 2) is None
    store.upsert(
        TrackletSnapshot(
            "camera",
            1,
            1,
            5,
            5,
            np.asarray([1, 0]),
            np.zeros((10, 10, 3), dtype=np.uint8),
        )
    )
    store.request_finalize()

    gallery = catalog.sync()

    assert gallery is not None
    assert gallery.match(np.asarray([1, 0])).key == "identity-0001"
    assert catalog.sync() is gallery
    catalog.close()


def test_tracklet_store_finalizes_and_applies_animal_number(tmp_path):
    database = tmp_path / "identities" / "cows.sqlite"
    preview = np.zeros((10, 10, 3), dtype=np.uint8)
    with TrackletStore(database, session="scan") as store:
        for source, track_id, embedding in (
            ("camera-1", 1, [1, 0]),
            ("camera-1", 2, [0, 1]),
            ("camera-2", 1, [0.99, 0.01]),
            ("camera-2", 2, [0.01, 0.99]),
        ):
            store.upsert(
                TrackletSnapshot(
                    source,
                    track_id,
                    1,
                    5,
                    5,
                    np.asarray(embedding),
                    preview,
                )
            )

        assignments = finalize_enrollment(store)
        identity = str(store.identities()[0]["identity"])
        store.set_animal_number(identity, "NL-123")
        _, keys, labels = store.gallery_data()

    assert len(set(assignments.values())) == 2
    assert identity in keys
    assert "NL-123" in labels


def test_tracklet_store_supports_detector_worker_thread(tmp_path):
    store = TrackletStore(tmp_path / "cows.sqlite", session="scan")
    snapshot = TrackletSnapshot(
        "camera-1",
        1,
        1,
        3,
        3,
        np.asarray([1, 0]),
        np.zeros((10, 10, 3), dtype=np.uint8),
    )

    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(store.upsert, snapshot).result()

    assert store.tracklets()[0].id.endswith(":camera-1:1")
    store.close()


def test_tracklet_store_records_and_validates_embedding_model(tmp_path):
    database = tmp_path / "cows.sqlite"
    preview = np.zeros((10, 10, 3), dtype=np.uint8)
    with TrackletStore(database, session="scan") as store:
        store.ensure_embedding_model("model-a", 2)
        store.upsert(
            TrackletSnapshot("camera", 1, 1, 5, 5, np.asarray([1, 0]), preview)
        )
        store.ensure_embedding_model("model-a", 2)

        assert store._control("embedding_model") == "model-a"
        assert store._control("embedding_dimension") == "2"
        with pytest.raises(ValueError, match="expected model-b"):
            store.ensure_embedding_model("model-b", 2)
        with pytest.raises(ValueError, match="incompatible embeddings"):
            store.ensure_embedding_model("model-a", 3)


def test_tracklet_store_updates_existing_identity_and_caps_learned_samples(tmp_path):
    preview = np.zeros((10, 10, 3), dtype=np.uint8)
    with TrackletStore(tmp_path / "cows.sqlite", session="scan") as store:
        store.upsert(
            TrackletSnapshot("camera-1", 1, 1, 1, 1, np.asarray([1, 0]), preview)
        )
        assignments = finalize_enrollment(store)
        identity = next(iter(assignments.values()))
        store.set_animal_number(identity, "NL-123")

        for track_id, embedding in enumerate(
            ([0.9, 0.4], [0.7, 0.7], [0.4, 0.9]), start=2
        ):
            store.update_identity(
                TrackletSnapshot(
                    "camera-1",
                    track_id,
                    1,
                    5,
                    5,
                    np.asarray(embedding),
                    preview,
                    identity,
                ),
                max_samples=2,
                duplicate_similarity=DEFAULT_REID_POLICY.duplicate_similarity,
            )

        learned = store.connection.execute(
            "SELECT COUNT(*) FROM tracklets WHERE learned = 1"
        ).fetchone()[0]
        _, keys, labels = store.gallery_data()

    assert learned == 2
    assert keys.count(identity) == 3
    assert set(labels) == {"NL-123"}


def test_tracklet_store_skips_duplicate_identity_sample(tmp_path):
    preview = np.zeros((10, 10, 3), dtype=np.uint8)
    with TrackletStore(tmp_path / "cows.sqlite", session="scan") as store:
        store.upsert(
            TrackletSnapshot("camera-1", 1, 1, 1, 1, np.asarray([1, 0]), preview)
        )
        identity = next(iter(finalize_enrollment(store).values()))

        added = store.update_identity(
            TrackletSnapshot(
                "camera-1",
                2,
                1,
                5,
                5,
                np.asarray([0.999, 0.001]),
                preview,
                identity,
            ),
            max_samples=DEFAULT_REID_POLICY.max_identity_samples,
            duplicate_similarity=DEFAULT_REID_POLICY.duplicate_similarity,
        )

    assert not added


def test_pending_enrollment_adds_identity_without_changing_existing_ones(tmp_path):
    preview = np.zeros((10, 10, 3), dtype=np.uint8)
    with TrackletStore(tmp_path / "cows.sqlite", session="scan") as store:
        for track_id, embedding in enumerate(([1, 0, 0], [0, 1, 0]), start=1):
            store.upsert(
                TrackletSnapshot(
                    "camera-1",
                    track_id,
                    1,
                    5,
                    5,
                    np.asarray(embedding),
                    preview,
                )
            )
        initial = finalize_enrollment(store, identity_count=2)
        first_identity = sorted(set(initial.values()))[0]
        store.set_animal_number(first_identity, "NL-123")

        for source, track_id, embedding in zip(
            ("camera-2", "camera-3", "camera-4"),
            range(3, 6),
            ([0, 0, 1], [0.2, 0, 0.98], [0, 0.2, 0.98]),
            strict=True,
        ):
            store.update_pending(
                TrackletSnapshot(
                    source,
                    track_id,
                    1,
                    5,
                    5,
                    np.asarray(embedding),
                    preview,
                ),
                max_samples=DEFAULT_REID_POLICY.max_pending_samples,
                duplicate_similarity=DEFAULT_REID_POLICY.duplicate_similarity,
            )

        added = finalize_pending_enrollment(
            store, similarity_threshold=0.95, margin_threshold=0
        )
        identities = store.identities()

    assert set(added.values()) == {"identity-0003"}
    assert len(identities) == 3
    assert (
        next(item for item in identities if item["identity"] == first_identity)[
            "animal_number"
        ]
        == "NL-123"
    )


def test_pending_enrollment_reuses_a_confident_existing_identity(tmp_path):
    preview = np.zeros((10, 10, 3), dtype=np.uint8)
    with TrackletStore(tmp_path / "cows.sqlite", session="scan") as store:
        for track_id, embedding in enumerate(([1, 0], [0, 1]), start=1):
            store.upsert(
                TrackletSnapshot(
                    "camera-1",
                    track_id,
                    1,
                    5,
                    5,
                    np.asarray(embedding),
                    preview,
                )
            )
        initial = finalize_enrollment(store, identity_count=2)
        first_identity = initial[next(iter(initial))]
        embeddings, keys, labels = store.gallery_data()
        gallery = IdentityGallery(
            embeddings,
            keys,
            labels,
            match_threshold=0.68,
            match_margin=0.05,
        )

        for source, track_id, angle in zip(
            ("camera-2", "camera-3", "camera-4"),
            range(3, 6),
            (10, 18, 26),
            strict=True,
        ):
            radians = np.deg2rad(angle)
            store.update_pending(
                TrackletSnapshot(
                    source,
                    track_id,
                    1,
                    5,
                    5,
                    np.asarray([np.cos(radians), np.sin(radians)]),
                    preview,
                ),
                max_samples=DEFAULT_REID_POLICY.max_pending_samples,
                duplicate_similarity=DEFAULT_REID_POLICY.duplicate_similarity,
            )

        assignments = finalize_pending_enrollment(
            store,
            similarity_threshold=0.95,
            margin_threshold=0,
            gallery=gallery,
        )
        identities = store.identities()
        learned = store.connection.execute(
            "SELECT COUNT(*) FROM tracklets WHERE learned = 1"
        ).fetchone()[0]

    assert set(assignments.values()) == {first_identity}
    assert len(identities) == 2
    assert learned == 3
    assert (
        next(
            identity["tracklets"]
            for identity in identities
            if identity["identity"] == first_identity
        )
        == 4
    )


def test_pending_enrollment_keeps_immature_candidates_pending(tmp_path):
    preview = np.zeros((10, 10, 3), dtype=np.uint8)
    with TrackletStore(tmp_path / "cows.sqlite", session="scan") as store:
        store.upsert(
            TrackletSnapshot("camera", 1, 1, 5, 5, np.asarray([1, 0]), preview)
        )
        finalize_enrollment(store)
        store.update_pending(
            TrackletSnapshot("camera", 2, 1, 5, 5, np.asarray([0, 1]), preview),
            max_samples=DEFAULT_REID_POLICY.max_pending_samples,
            duplicate_similarity=DEFAULT_REID_POLICY.duplicate_similarity,
        )

        with pytest.raises(ValueError, match="at least 3 tracklets"):
            finalize_pending_enrollment(store)

        assert len(store.pending_tracklets()) == 1


def test_pending_tracklets_deduplicate_different_tracks_but_update_same_track(tmp_path):
    preview = np.zeros((10, 10, 3), dtype=np.uint8)
    with TrackletStore(tmp_path / "cows.sqlite", session="scan") as store:
        first = TrackletSnapshot(
            "camera",
            1,
            1,
            5,
            5,
            np.asarray([1, 0]),
            preview,
        )
        duplicate = TrackletSnapshot(
            "camera",
            2,
            1,
            5,
            5,
            np.asarray([0.999, 0.001]),
            preview,
        )

        assert store.update_pending(
            first,
            max_samples=DEFAULT_REID_POLICY.max_pending_samples,
            duplicate_similarity=DEFAULT_REID_POLICY.duplicate_similarity,
        )
        assert store.update_pending(
            first,
            max_samples=DEFAULT_REID_POLICY.max_pending_samples,
            duplicate_similarity=DEFAULT_REID_POLICY.duplicate_similarity,
        )
        assert not store.update_pending(
            duplicate,
            max_samples=DEFAULT_REID_POLICY.max_pending_samples,
            duplicate_similarity=DEFAULT_REID_POLICY.duplicate_similarity,
        )
        pending = store.pending_tracklets()

    assert len(pending) == 1
    assert pending[0].track_id == 1


def test_public_dataset_adapters(tmp_path):
    multicam = tmp_path / "multicam" / "2023Aug14" / "007" / "frame.jpg"
    multicam.parent.mkdir(parents=True)
    multicam.touch()
    cows2021 = tmp_path / "cows2021" / "RGB" / "042" / "0" / "frame.jpg"
    cows2021.parent.mkdir(parents=True)
    cows2021.touch()

    multicam_samples = discover_public_dataset(
        f"multicamcows2024={tmp_path / 'multicam'}"
    )
    cows2021_samples = discover_public_dataset(f"cows2021={tmp_path / 'cows2021'}")

    assert multicam_samples[0].identity == "multicamcows2024:007"
    assert cows2021_samples[0].identity == "cows2021:042"


def test_video_benchmark_measures_identity_delay_per_track():
    gallery = IdentityGallery(
        np.asarray([[1, 0]], dtype=np.float32),
        ["001"],
        ["001"],
        match_threshold=0,
        match_margin=0.05,
    )
    observations = [
        VideoObservation(
            frame,
            [GroundTruth("001", (frame, 0, 10 + frame, 10))],
            [
                IdentityCandidate(
                    Crop(frame, 0, 10 + frame, 10, track_id=1),
                    np.zeros((10, 10, 3)),
                )
            ],
            np.asarray([[1, 0]], dtype=np.float32),
        )
        for frame in range(3)
    ]

    metrics = evaluate_observations(
        observations,
        gallery,
        sample_counts=[1, 2],
        track_max_age=2,
        ground_truth_iou=0.5,
    )

    assert metrics["1"]["identity_accuracy"] == 1
    assert metrics["2"]["identity_coverage"] == pytest.approx(2 / 3)
    assert metrics["2"]["mean_observations_to_identity"] == 1
