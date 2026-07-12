from datetime import datetime, timedelta

import numpy as np

from aidetector.detection.detector import Detector, ModelRuntime
from aidetector.detection.events import DetectionEvent, EventCollector
from aidetector.detection.models import Detection, ImageSet
from aidetector.detection.yolo import TrackedSourceResult
from aidetector.exporters.disk import DiskExporter
from aidetector.exporters.dispatcher import ExportDispatcher
from aidetector.exporters.factory import build_exporters
from aidetector.exporters.telegram import TelegramExporter
from aidetector.exporters.webhook import WebhookExporter
from aidetector.utils.config import (
    ChatConfig,
    DetectionConfig,
    DiskConfig,
    ExportersConfig,
    SSEConfig,
    WebhookConfig,
    YoloConfig,
)


def make_detection(
    date: datetime,
    confidence: dict[str, float] | None = None,
) -> Detection:
    image = np.zeros((80, 120, 3), dtype=np.uint8)
    return Detection(
        date,
        ImageSet(image),
        confidence if confidence is not None else {"cow": 0.9},
    )


class ImmediateExecutor:
    def submit(self, function, *args):
        function(*args)


class FakeValidator:
    def __init__(self, value=True):
        self.value = value
        self.calls = []

    def validate(self, event):
        self.calls.append(event)
        return self.value


class RecordingExporter:
    def __init__(self):
        self.calls = []
        self.closed = False

    def export(self, event, validated):
        self.calls.append((event, validated))

    def close(self):
        self.closed = True


class RecordingPublisher:
    def __init__(self):
        self.calls = []

    def publish_tracks(self, source, detection):
        self.calls.append((source, detection))


class FakeSourceProvider:
    def __init__(self, *, stream=True, batches=None, error=None):
        self.sources = ["camera-1", "camera-2"]
        self.stream = stream
        self.batches = batches or []
        self.error = error
        self.closed = False

    def is_stream(self):
        return self.stream

    def iter_batches(self):
        yield from self.batches
        if self.error:
            raise self.error

    def close(self):
        self.closed = True


class RecordingRunner:
    def __init__(self, mapped=None):
        self.calls = []
        self.mapped = mapped

    def detect(self, frames):
        self.calls.append(("detect", len(frames)))
        return [f"batch-{index}" for index in range(len(frames))]

    def track_sources(self, batch):
        self.calls.append(("track_sources", list(batch)))
        return [
            TrackedSourceResult(source, f"{source}-tracked", frames)
            for source, frames in batch.items()
        ]

    def detections_from_result(self, result, frames):
        if callable(self.mapped):
            return self.mapped(result, frames)
        return self.mapped


class RecordingDispatcher:
    def __init__(self):
        self.events = []
        self.closed = False

    def submit(self, event):
        self.events.append(event)

    def close(self):
        self.closed = True


def build_test_detector(
    config: YoloConfig,
    *,
    runner=None,
    source_provider=None,
    publisher=None,
):
    source_provider = source_provider or FakeSourceProvider()
    dispatcher = RecordingDispatcher()
    return (
        Detector(
            DetectionConfig(source=source_provider.sources),
            source_provider,
            ModelRuntime(
                config,
                runner or RecordingRunner(),
                EventCollector(config),
            ),
            dispatcher,
            [publisher] if publisher else [],
        ),
        dispatcher,
    )


def test_exporter_factory_builds_explicit_exporter_types(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    targets = build_exporters(
        ExportersConfig(
            disk=DiskConfig(directory="events"),
            webhook=WebhookConfig(url="https://example.test/hook"),
            telegram=ChatConfig(token="token", chat="chat-id"),
        ),
        detector_index=0,
    )

    assert [type(exporter) for exporter in targets.exporters] == [
        TelegramExporter,
        WebhookExporter,
        DiskExporter,
    ]
    assert targets.track_publishers == []


def test_exporter_factory_assigns_sse_endpoint_without_mutating_config(monkeypatch):
    created = []

    class RecordingSSEExporter:
        def __init__(self, config):
            self.config = config
            created.append(self)

        def publish_tracks(self, source, detection):
            pass

        def export(self, event, validated):
            pass

    monkeypatch.setattr(
        "aidetector.exporters.factory.SSEExporter",
        RecordingSSEExporter,
    )
    config = SSEConfig(port=9876)

    targets = build_exporters(ExportersConfig(sse=config), detector_index=2)

    assert config.endpoint is None
    assert created[0].config.endpoint == "/events/2"
    assert created[0].config.port == 9876
    assert targets.track_publishers == created


def test_event_collector_selects_best_complete_event():
    config = YoloConfig(model="model.pt", frames_min=2)
    collector = EventCollector(config)
    start = datetime(2026, 1, 1, 12, 0, 0)

    collector.add("camera", [make_detection(start, {"cow": 0.7})], now=start)
    collector.add(
        "camera",
        [make_detection(start + timedelta(seconds=1), {"cow": 0.9})],
        now=start + timedelta(seconds=1),
    )
    events = collector.flush_all()

    assert len(events) == 1
    assert len(events[0].detections) == 2
    assert events[0].best.confidence == {"cow": 0.9}


def test_event_collector_expires_and_keeps_trailing_frames():
    config = YoloConfig(
        model="model.pt",
        frames_min=1,
        timeout=5,
        include_trailing_time=2,
    )
    collector = EventCollector(config)
    start = datetime(2026, 1, 1, 12, 0, 0)
    collector.add("camera", [make_detection(start)], now=start)
    collector.add_trailing(
        "camera",
        [make_detection(start + timedelta(seconds=1), {})],
        now=start + timedelta(seconds=1),
    )

    events = collector.flush_expired(now=start + timedelta(seconds=6))

    assert len(events) == 1
    assert len(events[0].detections) == 2
    assert events[0].detections[-1].confidence == {}


def test_export_dispatcher_validates_and_exports_event():
    validator = FakeValidator(True)
    exporter = RecordingExporter()
    dispatcher = ExportDispatcher(
        validator,
        [exporter],
        YoloConfig(model="model.pt", confidence=0.5),
        executor=ImmediateExecutor(),
    )
    detection = make_detection(datetime.now())
    event = DetectionEvent("camera", (detection,), detection)

    dispatcher.submit(event)

    assert len(validator.calls) == 1
    assert exporter.calls == [(event, True)]


def test_export_dispatcher_serializes_cooldown_and_closes_exporters():
    validator = FakeValidator(True)
    exporter = RecordingExporter()
    dispatcher = ExportDispatcher(
        validator,
        [exporter],
        YoloConfig(model="model.pt", confidence=0.5, cooldown=60),
        executor=ImmediateExecutor(),
    )
    detection = make_detection(datetime.now())
    event = DetectionEvent("camera", (detection,), detection)

    dispatcher.submit(event)
    dispatcher.submit(event)
    dispatcher.close()

    assert len(validator.calls) == 1
    assert len(exporter.calls) == 1
    assert exporter.closed is True


def test_export_dispatcher_compares_cooldown_using_event_timestamps():
    validator = FakeValidator(True)
    exporter = RecordingExporter()
    dispatcher = ExportDispatcher(
        validator,
        [exporter],
        YoloConfig(model="model.pt", confidence=0.5, cooldown=60),
        executor=ImmediateExecutor(),
    )
    first = make_detection(datetime(2020, 1, 1, 12, 0, 0))
    second = make_detection(datetime(2020, 1, 1, 12, 0, 30))

    dispatcher.submit(DetectionEvent("camera", (first,), first))
    dispatcher.submit(DetectionEvent("camera", (second,), second))

    assert len(exporter.calls) == 1


def test_detector_batches_sources_when_tracking_is_disabled():
    config = YoloConfig(model="model.pt", tracking=False)
    runner = RecordingRunner()
    detector, _ = build_test_detector(config, runner=runner)
    handled = []
    detector._handle_model_result = lambda source, result, frames: handled.append(
        (source, result, len(frames))
    )
    frame = np.zeros((80, 120, 3), dtype=np.uint8)

    detector._handle_frame_batch(
        {
            "camera-1": [(datetime.now(), frame)],
            "camera-2": [(datetime.now(), frame)],
        }
    )

    assert runner.calls == [("detect", 2)]
    assert handled == [
        ("camera-1", "batch-0", 1),
        ("camera-2", "batch-1", 1),
    ]


def test_detector_tracks_sources_as_one_stream_batch():
    config = YoloConfig(model="model.pt", tracking=True)
    runner = RecordingRunner()
    detector, _ = build_test_detector(config, runner=runner)
    handled = []
    detector._handle_model_result = lambda source, result, frames: handled.append(
        (source, result, len(frames))
    )
    frame = np.zeros((80, 120, 3), dtype=np.uint8)

    detector._handle_frame_batch(
        {
            "camera-1": [(datetime.now(), frame), (datetime.now(), frame)],
            "camera-2": [(datetime.now(), frame)],
        }
    )

    assert runner.calls == [("track_sources", ["camera-1", "camera-2"])]
    assert handled == [
        ("camera-1", "camera-1-tracked", 2),
        ("camera-2", "camera-2-tracked", 1),
    ]


def test_detector_publishes_live_result_and_submits_completed_event():
    now = datetime.now()
    detections = [make_detection(now)]
    publisher = RecordingPublisher()
    runner = RecordingRunner(mapped=detections)
    config = YoloConfig(model="model.pt", frames_min=1, time_max=0)
    detector, dispatcher = build_test_detector(
        config,
        runner=runner,
        publisher=publisher,
    )

    detector._handle_model_result(
        "camera-1",
        object(),
        [(now, np.zeros((80, 120, 3), dtype=np.uint8))],
    )

    assert publisher.calls == [("0:0", detections[-1])]
    assert len(dispatcher.events) == 1


def test_detector_records_worker_failure_for_manager():
    source = FakeSourceProvider(error=ValueError("broken source"))
    config = YoloConfig(model="model.pt")
    detector, dispatcher = build_test_detector(config, source_provider=source)

    detector.start()
    detector.join(timeout=2)

    assert isinstance(detector.error, ValueError)
    assert source.closed is True
    assert dispatcher.closed is True


def test_detector_close_before_start_closes_dispatcher():
    config = YoloConfig(model="model.pt")
    detector, dispatcher = build_test_detector(config)

    detector.close()

    assert dispatcher.closed is True
