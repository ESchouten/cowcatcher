from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from aidetector.dazzlecow import benchmark
from aidetector.dazzlecow.benchmark import (
    open_set_metrics,
    parse_model,
    track_aggregation_metrics,
    write_manifest,
)
from aidetector.dazzlecow.calves8 import normalized_box, select_annotations
from aidetector.dazzlecow.folds import create_fold
from aidetector.dazzlecow.gallery import DazzleCowGallery, save_gallery
from aidetector.dazzlecow.metrics import clustering_metrics
from aidetector.dazzlecow.localizer import (
    CowCandidate,
    DazzleCowLocalizer,
    DazzleCowVideoLocalizer,
    LocalizerSettings,
    filtered_boxes,
    masked_candidate,
)
from aidetector.dazzlecow.runner import DazzleCowRunner
from aidetector.dazzlecow.tracks import TrackIdentityAggregator
from aidetector.dazzlecow.video_benchmark import (
    GroundTruth,
    VideoObservation,
    evaluate_observations,
)
from aidetector.dazzlecow.prepare import centered_candidate
from aidetector.dazzlecow.train import (
    CowImage,
    IdentityBatchSampler,
    TimestampBatchSampler,
    discover_public_dataset,
    leave_one_out_knn_accuracy,
    supervised_nt_xent,
)
from aidetector.utils.config import Crop, IdentityResult


def test_filtered_boxes_applies_area_filter_and_nms():
    outputs = [
        {"score": 0.9, "box": {"xmin": 10, "ymin": 10, "xmax": 50, "ymax": 50}},
        {"score": 0.8, "box": {"xmin": 12, "ymin": 12, "xmax": 52, "ymax": 52}},
        {"score": 0.7, "box": {"xmin": 70, "ymin": 70, "xmax": 75, "ymax": 75}},
    ]

    boxes, scores = filtered_boxes(outputs, (100, 100), 0.1, 0.3, 0.5)

    assert boxes == [[10, 10, 50, 50]]
    assert scores == [0.9]


def test_masked_candidate_uses_frame_dc_component_as_background():
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    frame[:] = [10, 20, 30]
    frame[1:3, 1:3] = [100, 110, 120]
    mask = np.zeros((4, 4), dtype=bool)
    mask[1:3, 1:3] = True

    candidate = masked_candidate(frame, mask, 0.9)

    assert candidate is not None
    assert candidate.crop == Crop(1, 1, 3, 3, label="cow", confidence=0.9)
    assert np.all(candidate.image == [100, 110, 120])


def test_localizer_returns_multiple_masked_cows():
    settings = LocalizerSettings(
        "owl",
        "sam",
        "cow",
        0.3,
        0.01,
        0.5,
        0.5,
        "cpu",
    )
    localizer = DazzleCowLocalizer.__new__(DazzleCowLocalizer)
    localizer.settings = settings
    localizer.owl = lambda *_args, **_kwargs: [
        {"score": 0.9, "box": {"xmin": 10, "ymin": 10, "xmax": 30, "ymax": 40}},
        {"score": 0.8, "box": {"xmin": 60, "ymin": 50, "xmax": 90, "ymax": 90}},
    ]
    masks = torch.zeros((2, 100, 100), dtype=torch.bool)
    masks[0, 10:40, 10:30] = True
    masks[1, 50:90, 60:90] = True
    result = type("Result", (), {"masks": type("Masks", (), {"data": masks})()})()
    localizer.sam = type("SAM", (), {"predict": lambda *_args, **_kwargs: [result]})()
    frame = np.full((100, 100, 3), 50, dtype=np.uint8)

    candidates = localizer.locate(frame)

    assert [candidate.crop for candidate in candidates] == [
        Crop(10, 10, 30, 40, label="cow", confidence=0.9),
        Crop(60, 50, 90, 90, label="cow", confidence=0.8),
    ]


def test_video_localizer_relocalizes_periodically_and_isolates_sources():
    class Predictor:
        def __init__(self):
            self.inference_state = {}
            self.calls = []

        def __call__(self, source, bboxes, stream):
            self.calls.append(
                (source.source, source.frame, bboxes, dict(self.inference_state))
            )
            self.inference_state = {"source": source.source, "frame": source.frame}
            masks = torch.ones((1, 4, 4), dtype=torch.bool)
            result = type(
                "Result",
                (),
                {"masks": type("Masks", (), {"data": masks})()},
            )()
            return iter([result])

    localizer = DazzleCowVideoLocalizer.__new__(DazzleCowVideoLocalizer)
    localizer.owl_interval = timedelta(seconds=1)
    localizer.last_owl = {}
    localizer.states = {}
    localizer.frames = {}
    localizer.scores = {}
    localizer.predictor = Predictor()
    box_calls = []
    localizer.boxes = lambda frame: (
        box_calls.append(frame.shape) or [[0, 0, 4, 4]],
        [0.9],
    )
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    start = datetime(2026, 1, 1)

    localizer.locate("camera-1", frame, start)
    localizer.locate("camera-1", frame, start + timedelta(seconds=0.5))
    localizer.locate("camera-2", frame, start + timedelta(seconds=0.5))
    localizer.locate("camera-1", frame, start + timedelta(seconds=1))

    assert len(box_calls) == 3
    assert [call[:3] for call in localizer.predictor.calls] == [
        ("camera-1", 0, [[0, 0, 4, 4]]),
        ("camera-1", 1, None),
        ("camera-2", 0, [[0, 0, 4, 4]]),
        ("camera-1", 0, [[0, 0, 4, 4]]),
    ]
    assert localizer.predictor.calls[1][3] == {"source": "camera-1", "frame": 0}
    assert localizer.predictor.calls[3][3] == {}


def test_gallery_uses_weighted_knn_and_threshold(tmp_path):
    path = tmp_path / "gallery.npz"
    save_gallery(
        path,
        np.array([[1, 0], [0.9, 0.1], [0, 1]], dtype=np.float32),
        ["001", "001", "002"],
    )
    gallery = DazzleCowGallery(path, neighbors=3, match_threshold=0.8)

    result = gallery.match(np.array([1, 0], dtype=np.float32))

    assert result is not None
    assert result.identity == "001"
    assert result.similarity == 1


def test_gallery_returns_none_below_threshold(tmp_path):
    path = tmp_path / "gallery.npz"
    save_gallery(path, np.array([[1, 0]], dtype=np.float32), ["001"])
    gallery = DazzleCowGallery(path, match_threshold=0.9)

    assert gallery.match(np.array([0, 1], dtype=np.float32)) is None


def test_gallery_returns_none_when_top_matches_are_too_close(tmp_path):
    path = tmp_path / "gallery.npz"
    save_gallery(
        path,
        np.array([[1, 0], [0.99, 0.01]], dtype=np.float32),
        ["001", "002"],
    )
    gallery = DazzleCowGallery(path, neighbors=2, match_threshold=0, match_margin=0.1)

    score = gallery.score(np.array([1, 0], dtype=np.float32))

    assert score.identity == "001"
    assert score.margin < 0.1
    assert gallery.match(np.array([1, 0], dtype=np.float32)) is None


def test_gallery_rejects_wrong_embedding_dimension(tmp_path):
    path = tmp_path / "gallery.npz"
    save_gallery(path, np.array([[1, 0]], dtype=np.float32), ["001"])
    gallery = DazzleCowGallery(path)

    with pytest.raises(ValueError, match="dimension does not match"):
        gallery.match(np.array([1, 0, 0], dtype=np.float32))


def test_runner_attaches_gallery_identity():
    candidate = CowCandidate(
        Crop(1, 2, 8, 9, label="cow", confidence=0.8),
        np.zeros((7, 7, 3), dtype=np.uint8),
    )
    runner = DazzleCowRunner.__new__(DazzleCowRunner)
    runner.localizer = type(
        "Localizer",
        (),
        {"locate": lambda _self, _source, _frame, _date: [candidate]},
    )()
    runner.encoder = type(
        "Encoder",
        (),
        {"embed": lambda _self, _images: np.array([[1, 0]], dtype=np.float32)},
    )()
    runner.gallery = type(
        "Gallery",
        (),
        {"match": lambda _self, _embedding: IdentityResult("cow-001", 0.95)},
    )()
    runner.identity_tracks = TrackIdentityAggregator(
        runner.gallery,
        samples=1,
        iou_threshold=0.2,
        max_age=10,
    )
    frame = np.zeros((10, 10, 3), dtype=np.uint8)

    result = runner.detect([frame], ["camera"])[0]
    detections = runner.detections_from_result(
        result,
        [(datetime(2026, 1, 1), frame)],
    )

    assert detections is not None
    assert detections[0].images.crops[0].track_id == 1
    assert detections[0].images.crops[0].identity == IdentityResult("cow-001", 0.95)


def test_track_identity_uses_aggregated_embeddings():
    calls = []

    class Gallery:
        def match(self, embedding):
            calls.append(embedding)
            return IdentityResult("cow-001", 0.9)

    tracks = TrackIdentityAggregator(
        Gallery(),
        samples=3,
        iou_threshold=0.2,
        max_age=10,
    )
    candidates = []
    for offset, embedding in enumerate(([1, 0], [0.8, 0.2], [1, 0])):
        candidate = CowCandidate(
            Crop(offset, 0, 10 + offset, 10, label="cow"),
            np.zeros((10, 10, 3), dtype=np.uint8),
        )
        tracks.apply("camera", [candidate], np.asarray([embedding], dtype=np.float32))
        candidates.append(candidate)

    assert [candidate.crop.track_id for candidate in candidates] == [1, 1, 1]
    assert [candidate.crop.identity for candidate in candidates[:2]] == [None, None]
    assert candidates[2].crop.identity == IdentityResult("cow-001", 0.9)
    assert len(calls) == 1
    assert np.linalg.norm(calls[0]) == pytest.approx(1)


def test_track_identity_separates_cows_and_expires_stale_tracks():
    gallery = type(
        "Gallery",
        (),
        {"match": lambda _self, _embedding: IdentityResult("cow", 1)},
    )()
    tracks = TrackIdentityAggregator(
        gallery,
        samples=1,
        iou_threshold=0.2,
        max_age=1,
    )
    left = CowCandidate(Crop(0, 0, 10, 10), np.zeros((10, 10, 3)))
    right = CowCandidate(Crop(20, 0, 30, 10), np.zeros((10, 10, 3)))
    tracks.apply("camera", [left, right], np.asarray([[1, 0], [0, 1]]))
    tracks.apply("camera", [], np.empty((0, 2), dtype=np.float32))
    replacement = CowCandidate(Crop(0, 0, 10, 10), np.zeros((10, 10, 3)))
    tracks.apply("camera", [replacement], np.asarray([[1, 0]]))

    assert [left.crop.track_id, right.crop.track_id] == [1, 2]
    assert replacement.crop.track_id == 3


def test_public_crop_preparation_selects_centered_cow():
    candidates = [
        CowCandidate(Crop(0, 0, 2, 2, confidence=0.9), np.zeros((2, 2, 3))),
        CowCandidate(Crop(4, 4, 6, 6, confidence=0.7), np.zeros((2, 2, 3))),
    ]

    selected = centered_candidate(candidates, (10, 10))

    assert selected is candidates[1]


def test_multicam_dataset_adapter_uses_parent_identity(tmp_path):
    image = tmp_path / "2023Aug14" / "007" / "frame.jpg"
    image.parent.mkdir(parents=True)
    image.touch()

    samples = discover_public_dataset(f"multicamcows2024={tmp_path}")

    assert samples[0].path == image
    assert samples[0].identity == "multicamcows2024:007"
    assert samples[0].group == "2023Aug14"


def test_cows2021_adapter_uses_first_numeric_identity(tmp_path):
    image = tmp_path / "RGB" / "042" / "0" / "frame.jpg"
    image.parent.mkdir(parents=True)
    image.touch()

    samples = discover_public_dataset(f"cows2021={tmp_path}")

    assert samples[0].identity == "cows2021:042"


def test_supervised_nt_xent_is_finite_and_differentiable():
    embeddings = torch.tensor(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]],
        requires_grad=True,
    )
    labels = torch.tensor([0, 0, 1, 1])

    loss = supervised_nt_xent(embeddings, labels)
    loss.backward()

    assert torch.isfinite(loss)
    assert embeddings.grad is not None


def test_identity_batch_sampler_selects_p_identities_and_k_images():
    samples = [
        CowImage(Path(f"{identity}-{index}.jpg"), identity)
        for identity in ("001", "002", "003")
        for index in range(3)
    ]
    sampler = IdentityBatchSampler(
        samples,
        identities_per_batch=2,
        images_per_identity=2,
        seed=1,
    )

    batch = next(iter(sampler))
    identities = [samples[index].identity for index in batch]

    assert len(batch) == 4
    assert sorted(Counter(identities).values()) == [2, 2]


def test_timestamp_sampler_groups_unique_cows_from_the_same_frame():
    samples = [
        CowImage(Path(f"{identity}-{timestamp}.jpg"), identity, timestamp)
        for timestamp in ("001", "002")
        for identity in ("cow-a", "cow-b")
    ]
    sampler = TimestampBatchSampler(samples, seed=1)

    batches = list(sampler)

    assert len(batches) == 2
    assert all(len(batch) == 2 for batch in batches)
    assert all(len({samples[index].identity for index in batch}) == 2 for batch in batches)


def test_leave_one_out_knn_accuracy():
    embeddings = torch.tensor(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]]
    )

    assert leave_one_out_knn_accuracy(embeddings, [0, 0, 1, 1], neighbors=1) == 1


def test_8_calves_annotations_are_split_chronologically():
    annotations = pd.DataFrame(
        [
            {
                "tracklet_id": identity,
                "frame_id": frame,
                "x": 0.5,
                "y": 0.5,
                "w": 0.25,
                "h": 0.5,
                "conf": 1.0,
            }
            for identity in (1, 2)
            for frame in range(1, 101)
        ]
    )

    selected = select_annotations(annotations, 10)

    assert selected.groupby(["tracklet_id", "split"]).size().to_dict() == {
        (1, "gallery"): 1,
        (1, "test"): 1,
        (1, "train"): 6,
        (1, "validation"): 2,
        (2, "gallery"): 1,
        (2, "test"): 1,
        (2, "train"): 6,
        (2, "validation"): 2,
    }
    assert selected[selected["split"] == "train"]["frame_id"].max() <= 50
    assert selected[selected["split"] == "validation"]["frame_id"].between(55, 65).all()
    assert selected[selected["split"] == "gallery"]["frame_id"].between(70, 80).all()
    assert selected[selected["split"] == "test"]["frame_id"].min() > 85


def test_8_calves_normalized_box_is_converted_to_pixels():
    row = type("Row", (), {"x": 0.5, "y": 0.5, "w": 0.25, "h": 0.5})()

    assert normalized_box(row, 800, 600) == [300, 150, 500, 450]


def test_benchmark_manifest_records_fixed_samples(tmp_path):
    image = tmp_path / "data" / "001" / "cow.jpg"
    image.parent.mkdir(parents=True)
    image.touch()
    output = tmp_path / "manifest.json"

    summary = write_manifest(tmp_path / "data", output)

    assert summary["count"] == 1
    assert summary["identities"] == 1
    assert '"path": "001/cow.jpg"' in output.read_text()


def test_benchmark_runs_models_on_the_same_splits(tmp_path, monkeypatch):
    sources = {}
    for split in ("train", "validation", "gallery", "test"):
        source = tmp_path / split
        image = source / "001" / "cow.jpg"
        image.parent.mkdir(parents=True)
        image.touch()
        sources[split] = source
    comparison = tmp_path / "previous.pt"
    comparison.touch()
    calls = []

    def fake_train(_datasets, output, **_settings):
        output.parent.mkdir(parents=True)
        output.touch()

    def fake_gallery(model, _source, output, **_settings):
        calls.append(("gallery", model.name))
        output.parent.mkdir(parents=True, exist_ok=True)
        save_gallery(output, np.asarray([[1, 0]]), ["001"])

    def fake_evaluate(model, *_args, **_settings):
        calls.append(("evaluate", model.name))
        return {"accuracy": 0.5, "coverage": 1.0}

    monkeypatch.setattr(benchmark, "train", fake_train)
    monkeypatch.setattr(benchmark, "build_gallery", fake_gallery)
    monkeypatch.setattr(benchmark, "evaluate", fake_evaluate)
    monkeypatch.setattr(benchmark, "encode_source", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        benchmark,
        "clustering_metrics",
        lambda *_args, **_kwargs: {"knn_accuracy": 0.5},
    )
    monkeypatch.setattr(
        benchmark,
        "track_aggregation_metrics",
        lambda *_args, **_kwargs: {"1": {"accuracy": 0.5}},
    )
    monkeypatch.setattr(
        benchmark,
        "open_set_metrics",
        lambda *_args, **_kwargs: (
            {"best_balanced_threshold": 0.8, "best_balanced_margin": 0.1},
            [
                {
                    "similarity_threshold": 0.8,
                    "margin_threshold": 0.1,
                    "known_identification_rate": 0.5,
                    "unknown_rejection_rate": 0.5,
                    "unknown_false_acceptance_rate": 0.5,
                    "balanced_accuracy": 0.5,
                }
            ],
        ),
    )

    report = benchmark.run_benchmark(
        train_source=sources["train"],
        validation_source=sources["validation"],
        gallery_source=sources["gallery"],
        test_source=sources["test"],
        output=tmp_path / "result",
        comparisons={"previous": comparison},
        epochs=1,
        identities_per_batch=1,
        images_per_identity=1,
        workers=0,
        learning_rate=0.001,
        temperature=0.5,
        patience=1,
        training_mode="supervised",
        batch_size=1,
        neighbors=1,
        match_threshold=0,
        match_margin=0,
        track_samples=[1],
        device="cpu",
        seed=1,
    )

    assert list(report["metrics"]) == ["trained", "previous"]
    assert calls == [
        ("gallery", "trained.pt"),
        ("evaluate", "trained.pt"),
        ("gallery", "previous.pt"),
        ("evaluate", "previous.pt"),
    ]
    assert (tmp_path / "result" / "benchmark.json").is_file()
    assert (tmp_path / "result" / "open-set" / "trained.csv").is_file()


def test_benchmark_model_specification():
    name, path = parse_model("previous=old.pt")

    assert name == "previous"
    assert path.name == "old.pt"

    with pytest.raises(ValueError, match="NAME=PATH"):
        parse_model("old.pt")


def test_benchmark_measures_rolling_track_aggregation():
    gallery = DazzleCowGallery.from_data(
        np.asarray([[1, 0], [0, 1]], dtype=np.float32),
        np.asarray(["001", "002"]),
        neighbors=1,
        match_threshold=0,
    )
    encoded = [
        (CowImage(Path(f"{identity}-{index}.jpg"), f"identity:{identity}"), embedding)
        for identity, embeddings in {
            "001": ([1, 0], [0.8, 0.2], [0.9, 0.1]),
            "002": ([0, 1], [0.2, 0.8], [0.1, 0.9]),
        }.items()
        for index, embedding in enumerate(embeddings)
    ]

    metrics = track_aggregation_metrics(encoded, gallery, [1, 2, 3])

    assert metrics["1"]["total"] == 6
    assert metrics["2"]["total"] == 4
    assert metrics["3"]["total"] == 2
    assert all(result["accuracy"] == 1 for result in metrics.values())


def test_benchmark_calibrates_unknown_rejection():
    gallery = DazzleCowGallery.from_data(
        np.asarray([[1, 0], [0, 1]], dtype=np.float32),
        np.asarray(["001", "002"]),
        neighbors=1,
        match_threshold=0,
    )
    encoded = [
        (CowImage(Path("001.jpg"), "identity:001"), np.asarray([1, 0])),
        (CowImage(Path("002.jpg"), "identity:002"), np.asarray([0, 1])),
    ]

    metrics, sweep = open_set_metrics(
        encoded,
        gallery,
        configured_threshold=0,
        configured_margin=0,
    )

    assert metrics["known_top1_accuracy"] == 1
    assert metrics["best_balanced"]["known_identification_rate"] == 1
    assert metrics["best_balanced"]["unknown_rejection_rate"] == 1
    assert 0 <= metrics["best_balanced_threshold"] <= 1
    assert 0 <= metrics["best_balanced_margin"] <= 1
    assert metrics["best_balanced_threshold"] or metrics["best_balanced_margin"]
    assert len(sweep) == 20301


def test_paper_clustering_metrics_match_separated_identities():
    metrics = clustering_metrics(
        np.asarray(
            [[1, 0], [0.99, 0.01], [0, 1], [0.01, 0.99]],
            dtype=np.float32,
        ),
        ["001", "001", "002", "002"],
        neighbors=1,
    )

    assert metrics == {
        "knn_accuracy": 1,
        "adjusted_rand_index": 1,
        "adjusted_mutual_info": 1,
        "normalized_mutual_info": 1,
        "hungarian_accuracy": 1,
    }


def test_group_folds_isolate_validation_and_test_data(tmp_path):
    samples = []
    for group in ("day-1", "day-2", "day-3"):
        for identity in ("001", "002"):
            path = tmp_path / "source" / group / identity / "cow.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
            samples.append(CowImage(path, f"identity:{identity}", "noon", group))

    report = create_fold(
        samples,
        tmp_path / "fold",
        validation_group="day-2",
        test_group="day-3",
    )

    assert report["counts"] == {"train": 2, "validation": 2, "test": 2}
    assert len(list((tmp_path / "fold" / "train").glob("*/*"))) == 2
    assert len(list((tmp_path / "fold" / "validation").glob("*/*"))) == 2
    assert len(list((tmp_path / "fold" / "gallery").glob("*/*"))) == 2
    assert len(list((tmp_path / "fold" / "test").glob("*/*"))) == 2


def test_video_benchmark_measures_identity_delay_per_track():
    gallery = DazzleCowGallery.from_data(
        np.asarray([[1, 0], [0, 1]], dtype=np.float32),
        np.asarray(["001", "002"]),
        neighbors=1,
        match_threshold=0,
    )
    observations = []
    for frame in range(3):
        observations.append(
            VideoObservation(
                frame,
                [
                    GroundTruth("001", (frame, 0, 10 + frame, 10)),
                    GroundTruth("002", (20 + frame, 0, 30 + frame, 10)),
                ],
                [
                    CowCandidate(
                        Crop(frame, 0, 10 + frame, 10),
                        np.zeros((10, 10, 3)),
                    ),
                    CowCandidate(
                        Crop(20 + frame, 0, 30 + frame, 10),
                        np.zeros((10, 10, 3)),
                    ),
                ],
                np.asarray([[1, 0], [0, 1]], dtype=np.float32),
            )
        )

    metrics = evaluate_observations(
        observations,
        gallery,
        sample_counts=[1, 2],
        track_iou=0.2,
        track_max_age=2,
        ground_truth_iou=0.5,
    )

    assert metrics["1"]["identity_accuracy"] == 1
    assert metrics["1"]["identity_coverage"] == 1
    assert metrics["1"]["track_id_switches"] == 0
    assert metrics["2"]["identity_coverage"] == pytest.approx(2 / 3)
    assert metrics["2"]["mean_observations_to_identity"] == 1
    assert metrics["2"]["tracks_without_identity"] == 0
