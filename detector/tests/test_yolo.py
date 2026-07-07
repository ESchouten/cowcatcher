from datetime import datetime, timedelta

import numpy as np

from aidetector.detection.yolo import YoloResultMapper, YoloRunner
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
    assert detections[0].images.crops == []
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
        return [f"{self.name}-tracking"]


def make_runner(tracking: bool) -> YoloRunner:
    runner = YoloRunner.__new__(YoloRunner)
    runner.config = YoloConfig(model="model.pt", tracking=tracking)
    runner.model = FakeModel("base")
    runner.tracking_models = {}
    runner.class_confidences = {19: ("cow", 0.5)}
    return runner


def test_yolo_runner_uses_predict_by_default():
    runner = make_runner(tracking=False)

    results = runner.predict([np.zeros((10, 10, 3), dtype=np.uint8)])

    assert results == ["base-prediction"]
    assert runner.model.calls[0][0] == "predict"
    assert "persist" not in runner.model.calls[0][2]


def test_yolo_runner_uses_persistent_tracking_when_enabled():
    runner = make_runner(tracking=True)

    results = runner.predict([np.zeros((10, 10, 3), dtype=np.uint8)], source="camera-1")

    assert results == ["base-tracking"]
    assert runner.model.calls[0][0] == "track"
    assert runner.model.calls[0][2]["persist"] is True


def test_yolo_runner_keeps_tracking_models_per_source():
    runner = make_runner(tracking=True)
    extra_models = [FakeModel("camera-2")]
    runner._new_model = extra_models.pop
    runner._setup_predictor = lambda _model: None
    frame = np.zeros((10, 10, 3), dtype=np.uint8)

    first = runner.predict([frame], source="camera-1")
    second = runner.predict([frame], source="camera-2")
    third = runner.predict([frame], source="camera-1")

    assert first == ["base-tracking"]
    assert second == ["camera-2-tracking"]
    assert third == ["base-tracking"]
    assert set(runner.tracking_models) == {"camera-1", "camera-2"}
    assert runner.tracking_models["camera-1"] is runner.model
    assert runner.tracking_models["camera-2"].name == "camera-2"
