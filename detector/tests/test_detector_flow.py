from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np

from aidetector.detection.detector import Detector
from aidetector.exporters.disk import DiskExporter
from aidetector.exporters.telegram import TelegramExporter
from aidetector.exporters.webhook import WebhookExporter
from aidetector.utils.config import (
    ChatConfig,
    Config,
    Detection,
    DetectionConfig,
    DetectorConfig,
    DiskConfig,
    ExportersConfig,
    ImageSet,
    OnnxConfig,
    WebhookConfig,
    YoloConfig,
)


def make_detection(date: datetime, confidence: dict[str, float] | None = None) -> Detection:
    image = np.zeros((80, 120, 3), dtype=np.uint8)
    return Detection(date, ImageSet(image), confidence or {"cow": 0.9})


class ImmediateExecutor:
    def submit(self, fn):
        fn()


class FakeValidator:
    def __init__(self, value):
        self.value = value

    def validate(self, best_detection, detections):
        return self.value


class RecordingExporter:
    def __init__(self):
        self.calls = []

    def export(self, best_detection, detections, validated):
        self.calls.append((best_detection, detections, validated))


def make_detector() -> Detector:
    detector = Detector.__new__(Detector)
    detector.detections = defaultdict(list)
    detector.yolo_config = None
    detector.validator = FakeValidator(True)
    detector.exporters = []
    detector.export_executor = ImmediateExecutor()
    detector.last_detection_time = {}
    return detector


def test_detector_from_config_builds_exporters():
    detector_config = DetectorConfig(
        detection=DetectionConfig(source=["video.mp4"]),
        exporters=ExportersConfig(
            disk=DiskConfig(directory="events"),
            webhook=WebhookConfig(url="https://example.test/hook"),
            telegram=ChatConfig(token="token", chat="chat-id"),
        ),
    )
    config = Config(detectors=[detector_config], onnx=OnnxConfig())

    detectors = Detector.from_config(config, detector_config)

    assert len(detectors) == 1
    assert [type(exporter) for exporter in detectors[0].exporters] == [
        TelegramExporter,
        WebhookExporter,
        DiskExporter,
    ]


def test_export_validates_exports_and_clears_detections():
    source = "camera"
    exporter = RecordingExporter()
    detector = make_detector()
    detector.exporters = [exporter]
    detector.detections[source] = [
        make_detection(datetime(2026, 1, 1, 12, 0, 0), {"cow": 0.7}),
        make_detection(datetime(2026, 1, 1, 12, 0, 2), {"cow": 0.9}),
    ]

    detector._export(source)

    assert detector.detections[source] == []
    assert len(exporter.calls) == 1
    best_detection, detections, validated = exporter.calls[0]
    assert best_detection.confidence == {"cow": 0.9}
    assert len(detections) == 2
    assert validated is True


def test_trailing_frames_are_included_after_latest_detection():
    source = "camera"
    detector = make_detector()
    detector.yolo_config = YoloConfig(
        model="model.pt",
        confidence=0.8,
        include_trailing_time=5,
    )
    detector.yolo_runner = type(
        "FakeYoloRunner",
        (),
        {"detections_from_result": lambda *_args, **_kwargs: None},
    )()
    detected_at = datetime(2026, 1, 1, 12, 0, 0)
    detector.detections[source] = [make_detection(detected_at)]
    processed = []
    detector._process = lambda src, detections=None: processed.append((src, detections))

    detector._handle_yolo_result(
        source,
        object(),
        [(detected_at + timedelta(seconds=2), np.zeros((80, 120, 3), dtype=np.uint8))],
    )

    assert processed[0][0] == source
    trailing = processed[0][1]
    assert len(trailing) == 1
    assert trailing[0].confidence == {}
