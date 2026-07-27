from datetime import datetime, timedelta

import numpy as np
import pytest

from aidetector.adapters.sinks.disk import DiskSink
from aidetector.adapters.sinks.factory import build_sinks
from aidetector.adapters.sinks.telegram import TelegramSink
from aidetector.adapters.sinks.webhook import WebhookSink
from aidetector.domain.events import DetectionEvent
from aidetector.domain.frames import Frame
from aidetector.domain.policies import CooldownPolicy, EventPolicy
from aidetector.media.artifacts import EventArtifacts
from aidetector.media.storage import compact_observation
from aidetector.pipeline.aggregation import EventAggregator
from aidetector.pipeline.cooldown import CooldownTracker
from aidetector.pipeline.dispatch import EventDispatcher
from aidetector.pipeline.inference import InferenceStage
from aidetector.pipeline.ports import ModelBatchResult
from aidetector.pipeline.processor import DetectionPipeline
from aidetector.utils.config import (
    ChatConfig,
    DiskConfig,
    ExportersConfig,
    SSEConfig,
    WebhookConfig,
)
from tests.factories import make_observation


def event_policy(**overrides) -> EventPolicy:
    values = {
        "frames_min": 1,
        "timeout": 5,
        "time_max": 60,
        "trailing_time": 1,
        **overrides,
    }
    return EventPolicy(**values)


class FakeValidator:
    def __init__(self, value=True):
        self.value = value
        self.calls = []

    def validate(self, event, artifacts):
        self.calls.append((event, artifacts))
        return self.value


class RecordingSink:
    def __init__(self):
        self.messages = []

    def send(self, message):
        self.messages.append(message)


class FakeSource:
    def __init__(self, *, realtime=True, batches=None, error=None):
        self.sources = ["camera-1", "camera-2"]
        self.realtime = realtime
        self.batches = batches or []
        self.error = error
        self.closed = False

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
            ModelBatchResult(source, f"{source}-tracked", frames)
            for source, frames in batch.items()
        ]

    def observations_from_result(self, result, frames):
        if callable(self.mapped):
            return self.mapped(result, frames)
        return self.mapped


def build_test_pipeline(
    *,
    tracking=False,
    runner=None,
    source=None,
    live_sink=None,
    identity_stage=None,
):
    source = source or FakeSource()
    dispatcher = RecordingDispatcher()
    pipeline = DetectionPipeline(
        interval=0,
        source=source,
        inference=InferenceStage(
            tracking,
            runner or RecordingRunner(),
            EventAggregator(event_policy(time_max=0)),
        ),
        identity_stage=identity_stage,
        dispatcher=dispatcher,
        live_sinks=[live_sink] if live_sink else [],
        compact=lambda observation: observation,
        resources=[],
    )
    return pipeline, dispatcher


class RecordingDispatcher:
    def __init__(self):
        self.events = []
        self.closed = False

    def submit(self, event):
        self.events.append(event)

    def start(self):
        pass

    def close(self):
        self.closed = True


def test_sink_factory_builds_explicit_adapter_types(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    bundle = build_sinks(
        ExportersConfig(
            disk=DiskConfig(directory="events"),
            webhook=WebhookConfig(url="https://example.test/hook"),
            telegram=ChatConfig(token="token", chat="chat-id"),
        ),
        pipeline_index=0,
    )

    assert [type(sink.target.target) for sink in bundle.events] == [
        TelegramSink,
        WebhookSink,
        DiskSink,
    ]
    assert bundle.live == []


def test_sink_factory_assigns_sse_endpoint_without_mutating_config(monkeypatch):
    created = []

    class RecordingSSESink:
        def __init__(self, config):
            self.config = config
            created.append(self)

        def send(self, message):
            pass

        def close(self):
            pass

    monkeypatch.setattr(
        "aidetector.adapters.sinks.factory.SSESink",
        RecordingSSESink,
    )
    config = SSEConfig(port=9876)

    bundle = build_sinks(ExportersConfig(sse=config), pipeline_index=2)

    assert config.endpoint is None
    assert created[0].config.endpoint == "/events/2"
    assert bundle.live == created
    assert bundle.resources[-1:] == created


def test_event_aggregator_selects_best_complete_event():
    aggregator = EventAggregator(event_policy(frames_min=2))
    start = datetime(2026, 1, 1, 12, 0, 0)
    aggregator.add("camera", [make_observation(start, {"cow": 0.7})], now=start)
    aggregator.add(
        "camera",
        [make_observation(start + timedelta(seconds=1), {"cow": 0.9})],
        now=start + timedelta(seconds=1),
    )

    events = aggregator.flush_all()

    assert len(events) == 1
    assert len(events[0].observations) == 2
    assert events[0].best.confidences == {"cow": 0.9}


def test_event_aggregator_expires_and_keeps_trailing_frames():
    aggregator = EventAggregator(event_policy(timeout=5, trailing_time=2))
    start = datetime(2026, 1, 1, 12, 0, 0)
    aggregator.add("camera", [make_observation(start)], now=start)
    aggregator.add_trailing(
        "camera",
        [make_observation(start + timedelta(seconds=1), {})],
        now=start + timedelta(seconds=1),
    )

    events = aggregator.flush_expired(now=start + timedelta(seconds=6))

    assert len(events) == 1
    assert len(events[0].observations) == 2
    assert events[0].observations[-1].confidences == {}


def test_event_aggregator_compacts_stored_frames():
    aggregator = EventAggregator(event_policy(), compact_observation)
    observation = make_observation(datetime.now())

    aggregator.add("camera", [observation])
    event = aggregator.flush_all()[0]

    assert observation.frame.image is not None
    assert event.best.frame.image is None
    assert event.best.frame.jpeg is not None


def test_observation_confidences_are_immutable():
    observation = make_observation(datetime.now())

    with pytest.raises(TypeError):
        observation.confidences["cow"] = 0.1  # type: ignore[index]


def test_dispatcher_validates_and_sends_completed_event():
    validator = FakeValidator(True)
    sink = RecordingSink()
    dispatcher = EventDispatcher(
        validator,
        [sink],
        CooldownTracker(CooldownPolicy(0.5, 0)),
        EventArtifacts,
    )
    dispatcher.start()
    observation = make_observation(datetime.now())
    event = DetectionEvent("camera", (observation,), observation)

    dispatcher.submit(event)
    dispatcher.close()

    assert validator.calls[0][0] is event
    assert sink.messages[0].event is event
    assert sink.messages[0].validated is True


def test_dispatcher_applies_cooldown_using_event_timestamps():
    validator = FakeValidator(True)
    sink = RecordingSink()
    dispatcher = EventDispatcher(
        validator,
        [sink],
        CooldownTracker(CooldownPolicy(0.5, 60)),
        EventArtifacts,
    )
    dispatcher.start()
    first = make_observation(datetime(2020, 1, 1, 12, 0, 0))
    second = make_observation(datetime(2020, 1, 1, 12, 0, 30))

    dispatcher.submit(DetectionEvent("camera", (first,), first))
    dispatcher.submit(DetectionEvent("camera", (second,), second))
    dispatcher.close()

    assert len(validator.calls) == 1
    assert len(sink.messages) == 1


def test_cooldown_only_records_eligible_classes():
    tracker = CooldownTracker(
        CooldownPolicy(
            {"cow": 0.5, "calf": 0.5},
            {"cow": 60, "calf": 0},
        )
    )
    first_at = datetime(2020, 1, 1, 12, 0, 0)
    first = make_observation(first_at, {"cow": 0.9, "calf": 0.9})
    first_event = DetectionEvent("camera", (first,), first)
    first_classes = tracker.eligible_classes(first_event)
    assert first_classes == ["cow", "calf"]
    tracker.record("camera", first_classes, first_at)

    second = make_observation(
        first_at + timedelta(seconds=1),
        {"cow": 0.9, "calf": 0.9},
    )
    second_event = DetectionEvent("camera", (second,), second)

    assert tracker.eligible_classes(second_event) == ["calf"]


def test_dispatcher_continues_after_event_processing_failure():
    attempts = []

    def artifact_factory(event):
        attempts.append(event)
        if len(attempts) == 1:
            raise ValueError("artifact failure")
        return EventArtifacts(event)

    sink = RecordingSink()
    dispatcher = EventDispatcher(
        FakeValidator(True),
        [sink],
        CooldownTracker(CooldownPolicy(0.5, 0)),
        artifact_factory,
    )
    dispatcher.start()
    first = make_observation(datetime(2020, 1, 1, 12, 0, 0))
    second = make_observation(datetime(2020, 1, 1, 12, 0, 1))

    dispatcher.submit(DetectionEvent("camera", (first,), first))
    dispatcher.submit(DetectionEvent("camera", (second,), second))
    dispatcher.close()

    assert len(attempts) == 2
    assert len(sink.messages) == 1
    assert sink.messages[0].event.best is second


def test_pipeline_batches_sources_when_tracking_is_disabled():
    runner = RecordingRunner()
    pipeline, _ = build_test_pipeline(runner=runner)
    handled = []
    pipeline.processor._handle_model_result = lambda source, result, frames: (
        handled.append((source, result, len(frames)))
    )
    frame = Frame(datetime.now(), np.zeros((80, 120, 3), dtype=np.uint8))

    pipeline.processor.process({"camera-1": [frame], "camera-2": [frame]})

    assert runner.calls == [("detect", 2)]
    assert handled == [("camera-1", "batch-0", 1), ("camera-2", "batch-1", 1)]


def test_pipeline_tracks_sources_as_one_stream_batch():
    runner = RecordingRunner()
    pipeline, _ = build_test_pipeline(tracking=True, runner=runner)
    handled = []
    pipeline.processor._handle_model_result = lambda source, result, frames: (
        handled.append((source, result, len(frames)))
    )
    frame = Frame(datetime.now(), np.zeros((80, 120, 3), dtype=np.uint8))

    pipeline.processor.process({"camera-1": [frame, frame], "camera-2": [frame]})

    assert runner.calls == [("track_sources", ["camera-1", "camera-2"])]
    assert handled == [
        ("camera-1", "camera-1-tracked", 2),
        ("camera-2", "camera-2-tracked", 1),
    ]


def test_identity_stage_runs_once_on_each_primary_tracking_result():
    now = datetime.now()
    mapped = [make_observation(now)]
    runner = RecordingRunner(mapped=mapped)

    class RecordingIdentityStage:
        def __init__(self):
            self.calls = []

        def enrich(self, source, observation):
            self.calls.append((source, observation))
            return observation

    identity_stage = RecordingIdentityStage()
    pipeline, _ = build_test_pipeline(
        tracking=True,
        runner=runner,
        identity_stage=identity_stage,
    )
    frame = Frame(now, np.zeros((80, 120, 3), dtype=np.uint8))

    pipeline.processor.process({"camera-1": [frame]})

    assert runner.calls == [("track_sources", ["camera-1"])]
    assert identity_stage.calls == [("camera-1", mapped[-1])]


def test_pipeline_publishes_live_observation_and_completed_event():
    now = datetime.now()
    observations = [make_observation(now)]
    live = RecordingSink()
    pipeline, dispatcher = build_test_pipeline(
        runner=RecordingRunner(mapped=observations),
        live_sink=live,
    )
    frame = Frame(now, np.zeros((80, 120, 3), dtype=np.uint8))

    pipeline.processor._handle_model_result("camera-1", object(), [frame])

    assert live.messages[0].source == "0:0"
    assert live.messages[0].observation is observations[-1]
    assert len(dispatcher.events) == 1


def test_pipeline_records_worker_failure_for_application():
    source = FakeSource(error=ValueError("broken source"))
    pipeline, dispatcher = build_test_pipeline(source=source)

    pipeline.start()
    pipeline.join(timeout=2)

    assert isinstance(pipeline.error, ValueError)
    assert source.closed is True
    assert dispatcher.closed is True


def test_pipeline_close_before_start_closes_dispatcher():
    pipeline, dispatcher = build_test_pipeline()

    pipeline.close()

    assert dispatcher.closed is True
