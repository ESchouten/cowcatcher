from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np

from aidetector.detection.detector import Detector
from aidetector.exporters.disk import DiskExporter
from aidetector.exporters.telegram import TelegramExporter
from aidetector.exporters.webhook import WebhookExporter
from aidetector.utils.config import (
    ChatConfig,
    Crop,
    DetectedObject,
    Config,
    Detection,
    DetectionConfig,
    DetectorConfig,
    DiskConfig,
    ExportersConfig,
    IdentityResult,
    ImageSet,
    OnnxConfig,
    SSEConfig,
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


class RecordingTracksExporter:
    def __init__(self):
        self.calls = []

    def publish_tracks(self, source, detection):
        self.calls.append((source, detection))


class CachedIdentityEnricher:
    def __init__(self):
        self.calls = []

    def enrich_live(self, source, detection):
        self.calls.append((source, detection))
        identity = IdentityResult(
            provider="cow-main",
            identity_id="cow-main-0001",
            name=None,
            status="matched",
            similarity=0.9,
        )
        detection.images.objects[0].identity = identity
        detection.identities = [identity]


class RecordingYoloRunner:
    def __init__(self):
        self.calls = []

    def predict(self, frames, source=None):
        self.calls.append((source, len(frames)))
        prefix = source or "batch"
        return [f"{prefix}-{index}" for index in range(len(frames))]


def make_detector() -> Detector:
    detector = Detector.__new__(Detector)
    detector.detections = defaultdict(list)
    detector.yolo_config = None
    detector.validator = FakeValidator(True)
    detector.exporters = []
    detector.identity_enricher = None
    detector.export_executor = ImmediateExecutor()
    detector.last_detection_time = {}
    detector.last_frame_time = datetime.min
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


def test_detector_from_config_uses_sse_config_port(monkeypatch):
    created = []

    class RecordingSSEExporter:
        def __init__(self, config):
            self.config = config
            created.append(self)

    monkeypatch.setattr(
        "aidetector.detection.detector.SSEExporter",
        RecordingSSEExporter,
    )
    detector_config = DetectorConfig(
        detection=DetectionConfig(source=["video.mp4"]),
        exporters=ExportersConfig(
            sse=[
                SSEConfig(endpoint="/events", port=9876),
                SSEConfig(endpoint="/events", port=9876),
            ],
        ),
    )
    config = Config(
        detectors=[detector_config],
        onnx=OnnxConfig(),
    )

    detectors = Detector.from_config(config, detector_config)

    assert detectors[0].exporters == created
    assert [exporter.config.port for exporter in created] == [9876, 9876]


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


def test_tracks_are_published_for_latest_yolo_detection():
    source = "camera"
    detected_at = datetime(2026, 1, 1, 12, 0, 0)
    published = RecordingTracksExporter()
    detector = make_detector()
    detector.yolo_config = YoloConfig(model="model.pt", confidence=0.8)
    detections = [
        Detection(
            detected_at,
            ImageSet(np.zeros((80, 120, 3), dtype=np.uint8)),
            {},
        ),
        make_detection(detected_at + timedelta(seconds=1)),
    ]
    detector.yolo_runner = type(
        "FakeYoloRunner",
        (),
        {"detections_from_result": lambda *_args, **_kwargs: detections},
    )()
    detector.exporters = [published]
    processed = []
    detector._process = lambda src, items=None: processed.append((src, items))

    detector._handle_yolo_result(
        source,
        object(),
        [(detected_at, np.zeros((80, 120, 3), dtype=np.uint8))],
    )

    assert published.calls == [(source, detections[-1])]
    assert processed == [(source, detections)]


def test_tracks_are_live_enriched_before_publishing():
    source = "camera"
    detected_at = datetime(2026, 1, 1, 12, 0, 0)
    published = RecordingTracksExporter()
    detector = make_detector()
    detector.yolo_config = YoloConfig(model="model.pt", confidence=0.8)
    detection = Detection(
        detected_at,
        ImageSet(
            np.zeros((80, 120, 3), dtype=np.uint8),
            [DetectedObject(Crop(10, 10, 40, 40, label="cow", confidence=0.9), track_id=7)],
        ),
        {"cow": 0.9},
    )
    detector.yolo_runner = type(
        "FakeYoloRunner",
        (),
        {"detections_from_result": lambda *_args, **_kwargs: [detection]},
    )()
    identity_enricher = CachedIdentityEnricher()
    detector.identity_enricher = identity_enricher
    detector.exporters = [published]
    detector._process = lambda *_args, **_kwargs: None

    detector._handle_yolo_result(
        source,
        object(),
        [(detected_at, np.zeros((80, 120, 3), dtype=np.uint8))],
    )

    published_detection = published.calls[0][1]
    assert identity_enricher.calls == [(source, detection)]
    assert published_detection.images.objects[0].identity.identity_id == "cow-main-0001"
    assert published_detection.identities[0].identity_id == "cow-main-0001"


def test_empty_tracks_are_published_when_yolo_has_no_detection():
    source = "camera"
    detected_at = datetime(2026, 1, 1, 12, 0, 0)
    published = RecordingTracksExporter()
    detector = make_detector()
    detector.yolo_config = YoloConfig(model="model.pt", confidence=0.8)
    detector.yolo_runner = type(
        "FakeYoloRunner",
        (),
        {"detections_from_result": lambda *_args, **_kwargs: None},
    )()
    detector.exporters = [published]

    detector._handle_yolo_result(
        source,
        object(),
        [(detected_at, np.zeros((80, 120, 3), dtype=np.uint8))],
    )

    assert len(published.calls) == 1
    assert published.calls[0][0] == source
    assert published.calls[0][1].confidence == {}
    assert published.calls[0][1].images.objects == []


def test_detector_batches_sources_when_tracking_is_disabled():
    detector = make_detector()
    detector.detection = DetectionConfig(source=["camera-1", "camera-2"])
    detector.yolo_config = YoloConfig(model="model.pt", tracking=False)
    detector.yolo_runner = RecordingYoloRunner()
    handled = []
    detector._handle_yolo_result = (
        lambda source, result, frames: handled.append((source, result, len(frames)))
    )
    frame = np.zeros((80, 120, 3), dtype=np.uint8)

    detector._handle_frame_batch(
        {
            "camera-1": [(datetime(2026, 1, 1, 12, 0, 0), frame)],
            "camera-2": [(datetime(2026, 1, 1, 12, 0, 1), frame)],
        }
    )

    assert detector.yolo_runner.calls == [(None, 2)]
    assert handled == [
        ("camera-1", "batch-0", 1),
        ("camera-2", "batch-1", 1),
    ]


def test_detector_tracks_each_source_independently_when_tracking_is_enabled():
    detector = make_detector()
    detector.detection = DetectionConfig(source=["camera-1", "camera-2"])
    detector.yolo_config = YoloConfig(model="model.pt", tracking=True)
    detector.yolo_runner = RecordingYoloRunner()
    handled = []
    detector._handle_yolo_result = (
        lambda source, result, frames: handled.append((source, result, len(frames)))
    )
    frame = np.zeros((80, 120, 3), dtype=np.uint8)

    detector._handle_frame_batch(
        {
            "camera-1": [
                (datetime(2026, 1, 1, 12, 0, 0), frame),
                (datetime(2026, 1, 1, 12, 0, 1), frame),
            ],
            "camera-2": [(datetime(2026, 1, 1, 12, 0, 2), frame)],
        }
    )

    assert detector.yolo_runner.calls == [("camera-1", 1), ("camera-2", 1)]
    assert handled == [
        ("camera-1", "camera-1-0", 2),
        ("camera-2", "camera-2-0", 1),
    ]
