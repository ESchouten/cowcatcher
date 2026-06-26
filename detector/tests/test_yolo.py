from datetime import datetime, timedelta

import numpy as np

from aidetector.detection.yolo import YoloResultMapper


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
