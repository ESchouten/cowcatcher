from datetime import datetime, timedelta

import numpy as np
import pytest

from aidetector.detection.yolo import (
    UltralyticsStreamBatch,
    YoloResultMapper,
    YoloRunner,
)
from aidetector.utils.config import YoloConfig


class FakeScalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class FakeBox:
    def __init__(self, class_id, confidence, xyxy):
        self.cls = FakeScalar(class_id)
        self.conf = FakeScalar(confidence)
        self.xyxy = [xyxy]


class FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


def test_yolo_result_mapper_keeps_all_boxes_above_threshold():
    mapper = YoloResultMapper(
        {
            0: ("cow", 0.5),
            1: ("horse", 0.7),
        }
    )
    result = FakeResult(
        [
            FakeBox(0, 0.8, [10, 20, 30, 40]),
            FakeBox(0, 0.6, [50, 60, 70, 80]),
            FakeBox(1, 0.6, [1, 2, 3, 4]),
        ]
    )
    start = datetime(2026, 1, 1, 12, 0, 0)
    frames = [
        (start, np.zeros((100, 100, 3), dtype=np.uint8)),
        (start + timedelta(seconds=1), np.zeros((100, 100, 3), dtype=np.uint8)),
    ]

    detections = mapper.detections_from_result(result, frames)

    assert detections is not None
    assert len(detections) == 2
    assert detections[0].confidence == {}
    assert detections[1].confidence == {"cow": 0.8}
    assert len(detections[1].images.crops) == 2
    assert detections[1].images.crops[0].label == "cow"


class FakeModel:
    def __init__(self, name="model"):
        self.name = name
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(("predict", self.name, kwargs))
        return [f"{self.name}-prediction"]

    def track(self, **kwargs):
        self.calls.append(("track", self.name, kwargs))
        if isinstance(kwargs.get("source"), UltralyticsStreamBatch):
            return [
                f"{self.name}-tracking-{source}" for source in kwargs["source"].sources
            ]
        return [f"{self.name}-tracking"]


def make_runner() -> YoloRunner:
    runner = YoloRunner.__new__(YoloRunner)
    runner.config = YoloConfig(model="model.pt")
    runner.model = FakeModel("base")
    runner.sources = ["camera-1", "camera-2"]
    runner.tracking_last_frames = {}
    runner.class_confidences = {19: ("cow", 0.5)}
    return runner


def test_ultralytics_stream_batch_behaves_like_stream_loader():
    frame_a = np.zeros((10, 10, 3), dtype=np.uint8)
    frame_b = np.ones((10, 10, 3), dtype=np.uint8)
    batch = UltralyticsStreamBatch(["camera-1", "camera-2"], [frame_a, frame_b])

    paths, images, messages = next(iter(batch))

    assert batch.mode == "stream"
    assert batch.source_type.stream is True
    assert len(batch) == 2
    assert paths == ["camera-1", "camera-2"]
    assert images[0] is frame_a
    assert images[1] is frame_b
    assert messages == ["", ""]
    with pytest.raises(StopIteration):
        next(batch)


def test_yolo_runner_detects_frames():
    runner = make_runner()

    results = runner.detect([np.zeros((10, 10, 3), dtype=np.uint8)])

    assert results == ["base-prediction"]
    assert runner.model.calls[0][0] == "predict"
    assert "persist" not in runner.model.calls[0][2]


def test_yolo_runner_tracks_latest_sources_as_stream_batch():
    runner = make_runner()
    detected_at = datetime(2026, 1, 1, 12, 0, 0)
    frame_1 = np.full((10, 10, 3), 1, dtype=np.uint8)
    frame_2 = np.full((10, 10, 3), 2, dtype=np.uint8)

    first = runner.track_sources(
        {"camera-1": [(detected_at, frame_1)]},
    )
    second = runner.track_sources(
        {"camera-2": [(detected_at + timedelta(seconds=1), frame_2)]},
    )

    assert [(item.source, item.result) for item in first] == [
        ("camera-1", "base-tracking-source-0")
    ]
    assert [(item.source, item.result) for item in second] == [
        ("camera-2", "base-tracking-source-1")
    ]
    first_source = runner.model.calls[0][2]["source"]
    second_source = runner.model.calls[1][2]["source"]
    assert isinstance(first_source, UltralyticsStreamBatch)
    assert first_source.sources == ["source-0", "source-1"]
    assert runner.model.calls[0][2]["batch"] == 2
    assert second_source.sources == ["source-0", "source-1"]
    assert second_source.images[0] is frame_1
    assert second_source.images[1] is frame_2
