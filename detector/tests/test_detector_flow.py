from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np

from aidetector.detection.detector import Detector
from aidetector.detection.identity_registry import IdentityRegistry
from aidetector.detection.manager import Manager
from aidetector.detection.yolo import TrackedSourceResult
from aidetector.exporters.disk import DiskExporter
from aidetector.exporters.telegram import TelegramExporter
from aidetector.exporters.webhook import WebhookExporter
from aidetector.utils.config import (
    ChatConfig,
    Config,
    Crop,
    Detection,
    DetectionConfig,
    DetectorConfig,
    DiskConfig,
    ExportersConfig,
    ImageSet,
    IdentityResult,
    OnnxConfig,
    SSEConfig,
    WebhookConfig,
    YoloConfig,
)


def make_detection(
    date: datetime, confidence: dict[str, float] | None = None
) -> Detection:
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


class RecordingYoloRunner:
    def __init__(self):
        self.calls = []

    def detect(self, frames):
        self.calls.append(("detect", len(frames)))
        return [f"batch-{index}" for index in range(len(frames))]

    def track_sources(self, batch):
        self.calls.append(("track_sources", list(batch)))
        return [
            TrackedSourceResult(source, f"{source}-tracked", frames)
            for source, frames in batch.items()
        ]


def make_detector() -> Detector:
    detector = Detector.__new__(Detector)
    detector.detections = defaultdict(list)
    detector.yolo_config = None
    detector.yolo_runner = None
    detector.identity_config = None
    detector.identity_pipeline = None
    detector.identity_registry = IdentityRegistry()
    detector.identity_producer = "test-detector"
    detector.validator = FakeValidator(True)
    detector.exporters = []
    detector.export_executor = ImmediateExecutor()
    detector.last_detection_time = {}
    detector.last_frame_time = datetime.min
    detector.last_identity_result_time = {}
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


def test_detector_from_config_builds_sse_exporter(monkeypatch):
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
        exporters=ExportersConfig(sse=SSEConfig(endpoint="/events", port=9876)),
    )
    config = Config(detectors=[detector_config], onnx=OnnxConfig())

    detectors = Detector.from_config(config, detector_config)

    assert detectors[0].exporters == created
    assert created[0].config.port == 9876


def test_detector_from_config_defaults_sse_endpoint_to_detector_index(monkeypatch):
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
        exporters=ExportersConfig(sse=SSEConfig()),
    )
    config = Config(detectors=[detector_config], onnx=OnnxConfig())

    Detector.from_config(config, detector_config, detector_index=2)

    assert created[0].config.endpoint == "/events/2"


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

    detector._handle_model_result(
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

    detector._handle_model_result(
        source,
        object(),
        [(detected_at, np.zeros((80, 120, 3), dtype=np.uint8))],
    )

    assert published.calls == [(source, detections[-1])]
    assert processed == [(source, detections)]


def test_detector_uses_identities_published_by_another_detector():
    source = "camera"
    detected_at = datetime(2026, 1, 1, 12, 0, 0)
    detector = make_detector()
    detector.yolo_config = YoloConfig(model="model.pt", confidence=0.8)
    identity_detection = Detection(
        detected_at,
        ImageSet(
            np.zeros((80, 120, 3), dtype=np.uint8),
            [
                Crop(
                    10,
                    10,
                    30,
                    30,
                    track_id=7,
                    identities=[IdentityResult("NL-123", 0.95)],
                )
            ],
        ),
        {},
    )
    event = Detection(
        detected_at,
        ImageSet(
            np.zeros((160, 240, 3), dtype=np.uint8),
            [Crop(0, 0, 80, 80, label="mounting")],
        ),
        {"mounting": 0.9},
    )
    detector.identity_registry.publish(source, "identity-detector", identity_detection)
    detector.yolo_runner = type(
        "FakeYoloRunner",
        (),
        {"detections_from_result": lambda *_args, **_kwargs: [event]},
    )()
    detector._process = lambda *_args, **_kwargs: None

    detector._handle_model_result(
        source,
        object(),
        [(detected_at, np.zeros((160, 240, 3), dtype=np.uint8))],
    )

    assert event.identities == [IdentityResult("NL-123", 0.95)]


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

    detector._handle_model_result(
        source,
        object(),
        [(detected_at, np.zeros((80, 120, 3), dtype=np.uint8))],
    )

    assert len(published.calls) == 1
    assert published.calls[0][0] == source
    assert published.calls[0][1].confidence == {}
    assert published.calls[0][1].images.crops == []


def test_detector_batches_sources_when_tracking_is_disabled():
    detector = make_detector()
    detector.detection = DetectionConfig(source=["camera-1", "camera-2"])
    detector.yolo_config = YoloConfig(model="model.pt", tracking=False)
    detector.yolo_runner = RecordingYoloRunner()
    handled = []
    detector._handle_model_result = lambda source, result, frames: handled.append(
        (source, result, len(frames))
    )
    frame = np.zeros((80, 120, 3), dtype=np.uint8)

    detector._handle_frame_batch(
        {
            "camera-1": [(datetime(2026, 1, 1, 12, 0, 0), frame)],
            "camera-2": [(datetime(2026, 1, 1, 12, 0, 1), frame)],
        }
    )

    assert detector.yolo_runner.calls == [("detect", 2)]
    assert handled == [
        ("camera-1", "batch-0", 1),
        ("camera-2", "batch-1", 1),
    ]


def test_detector_tracks_sources_as_stream_batch_when_tracking_is_enabled():
    detector = make_detector()
    detector.detection = DetectionConfig(source=["camera-1", "camera-2"])
    detector.yolo_config = YoloConfig(model="model.pt", tracking=True)
    detector.yolo_runner = RecordingYoloRunner()
    handled = []
    detector._handle_model_result = lambda source, result, frames: handled.append(
        (source, result, len(frames))
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

    assert detector.yolo_runner.calls == [("track_sources", ["camera-1", "camera-2"])]
    assert handled == [
        ("camera-1", "camera-1-tracked", 2),
        ("camera-2", "camera-2-tracked", 1),
    ]


class RecordingIdentityYoloRunner:
    def __init__(self):
        self.calls = []

    def track_sources(self, batch):
        self.calls.append({source: len(frames) for source, frames in batch.items()})
        return [
            TrackedSourceResult(
                source,
                [f"cow-{len(frames) - 1}"],
                frames,
            )
            for source, frames in batch.items()
        ]

    def detections_from_result(self, result, frames):
        return [make_detection(frames[-1][0], {result[0]: 0.9})]


class RecordingIdentityPipeline:
    reuses_primary_yolo = True

    def candidates_from_primary(self, _source, _result, _frame):
        return []

    def live_detection(self, date, frame, _candidates):
        return Detection(date, ImageSet(frame), {})


def test_identity_tracks_every_batch_but_processes_at_detection_interval():
    source = "camera"
    detector = make_detector()
    detector.detection = DetectionConfig(source=[source], interval=1)
    detector.yolo_config = YoloConfig(model="yolo26m-seg.pt", tracking=True)
    detector.yolo_runner = RecordingIdentityYoloRunner()
    detector.identity_pipeline = RecordingIdentityPipeline()
    handled = []
    published = RecordingTracksExporter()
    detector.exporters = [published]
    detector._handle_model_result = lambda item_source, result, frames: handled.append(
        (item_source, result, len(frames))
    )
    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    start = datetime(2026, 1, 1, 12, 0, 0)

    detector._handle_frame_batch(
        {
            source: [
                (start, frame),
                (start + timedelta(seconds=0.2), frame),
            ]
        }
    )
    detector._handle_frame_batch({source: [(start + timedelta(seconds=0.4), frame)]})
    detector._handle_frame_batch({source: [(start + timedelta(seconds=1.2), frame)]})

    assert detector.yolo_runner.calls == [
        {source: 2},
        {source: 1},
        {source: 1},
    ]
    assert handled == [
        (source, ["cow-1"], 2),
        (source, ["cow-0"], 1),
    ]
    assert len(published.calls) == 1
    assert published.calls[0][0] == source
    assert published.calls[0][1].confidence == {"cow-0": 0.9}


def test_manager_shares_one_identity_registry_between_detectors():
    config = Config(
        detectors=[
            DetectorConfig(detection=DetectionConfig(source=["camera"])),
            DetectorConfig(detection=DetectionConfig(source=["camera"])),
        ],
        onnx=OnnxConfig(),
    )

    manager = Manager.from_config(config)

    assert all(
        detector.identity_registry is manager.identity_registry
        for detector in manager.detectors
    )
