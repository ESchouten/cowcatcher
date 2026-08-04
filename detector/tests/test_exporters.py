import json
from datetime import datetime, timedelta
from pathlib import Path
from threading import Event

from aidetector.adapters.sinks.disk import DiskSink
from aidetector.adapters.sinks.media import EncodedFile
from aidetector.adapters.sinks.metadata import DetectionMetadata
from aidetector.adapters.sinks.sse import SSESink, tracks_payload
from aidetector.adapters.sinks.sse_transport import SSEHub
from aidetector.adapters.sinks.telegram import TelegramSink
from aidetector.adapters.sinks.webhook import WebhookSink
from aidetector.domain.detections import DetectedObject, Observation
from aidetector.domain.events import DetectionEvent, LiveObservation
from aidetector.domain.identity import IdentityResult
from aidetector.domain.policies import ExportPolicy
from aidetector.media.artifacts import EventArtifacts
from aidetector.pipeline.messages import CompletedEvent
from aidetector.pipeline.sinks import BufferedSink, FilteredEventSink
from aidetector.utils.config import (
    ChatConfig,
    DiskConfig,
    SSEConfig,
    WebhookConfig,
)
from tests.factories import make_event as build_event
from tests.factories import make_observation


def make_observations() -> list[Observation]:
    start = datetime(2026, 1, 1, 12, 0, 0)
    return [
        make_observation(
            start,
            {"cow": 0.7},
            objects=(DetectedObject(10, 10, 40, 50, "cow", 0.7),),
        ),
        make_observation(
            start + timedelta(seconds=2),
            {"cow": 0.9},
            objects=(DetectedObject(12, 12, 42, 52, "cow", 0.9),),
        ),
    ]


def make_event(observations: list[Observation] | None = None) -> DetectionEvent:
    return build_event(observations or make_observations())


def completed(
    event: DetectionEvent | None = None,
    validated: bool | None = True,
) -> CompletedEvent:
    event = event or make_event()
    return CompletedEvent(event, validated, EventArtifacts(event))


class RecordingSink:
    def __init__(self):
        self.messages = []

    def send(self, message):
        self.messages.append(message)


class BlockingSink(RecordingSink):
    def __init__(self):
        super().__init__()
        self.started = Event()
        self.release = Event()

    def send(self, message):
        self.started.set()
        self.release.wait()
        super().send(message)


def test_filtered_sink_applies_confidence_and_rejected_policy():
    message = completed()

    target = RecordingSink()
    FilteredEventSink(ExportPolicy(0.95, False), target).send(message)
    assert target.messages == []

    target = RecordingSink()
    FilteredEventSink(ExportPolicy(0.5, False), target).send(completed(validated=False))
    assert target.messages == []

    target = RecordingSink()
    FilteredEventSink(ExportPolicy(0.5, True), target).send(completed(validated=False))
    assert len(target.messages) == 1


def test_event_artifacts_cache_equal_requests(monkeypatch):
    calls = []

    def frame_jpg(_frame):
        calls.append(1)
        return b"jpg"

    monkeypatch.setattr("aidetector.media.artifacts.frame_jpg", frame_jpg)
    artifacts = EventArtifacts(make_event())

    assert artifacts.image() == b"jpg"
    assert artifacts.image() == b"jpg"
    assert calls == [1]


def test_buffered_sink_keeps_processing_latest_events():
    target = BlockingSink()
    sink = BufferedSink(target)
    messages = [completed() for _ in range(4)]
    sink.start()

    sink.send(messages[0])
    assert target.started.wait(1)
    for message in messages[1:]:
        sink.send(message)
    target.release.set()
    sink.close()

    assert target.messages == [messages[0], messages[2], messages[3]]


def test_disk_sink_writes_detection_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "aidetector.media.artifacts.generate_mp4",
        lambda *_args, **_kwargs: b"mp4",
    )
    message = completed()

    DiskSink(DiskConfig(directory=Path("events"))).send(message)

    event_dirs = list((tmp_path / "detections" / "events" / "approved").iterdir())
    assert len(event_dirs) == 1
    event_dir = event_dirs[0]
    assert event_dir.name.endswith(message.event.event_id)
    assert (event_dir / "best.jpg").exists()
    assert (event_dir / "clean.jpg").exists()
    assert (event_dir / "video.mp4").read_bytes() == b"mp4"

    metadata = json.loads((event_dir / "metadata.json").read_text())
    assert metadata["validated"] is True
    assert metadata["event_id"] == message.event.event_id
    assert metadata["source"] == "camera"
    assert metadata["timestamp"] == message.event.best.frame.captured_at.isoformat()
    assert metadata["confidence"] == 0.9
    assert metadata["observations"] == 2
    assert metadata["crop"] == {
        "x1": 12,
        "y1": 12,
        "x2": 42,
        "y2": 52,
        "label": "cow",
        "confidence": 0.9,
        "track_id": None,
        "identity": None,
    }


def test_disk_sink_uses_unique_directory_per_event(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "aidetector.media.artifacts.generate_mp4",
        lambda *_args, **_kwargs: None,
    )
    sink = DiskSink(DiskConfig(directory=Path("events")))

    sink.send(completed())
    sink.send(completed())

    event_dirs = list((tmp_path / "detections" / "events" / "approved").iterdir())
    assert len(event_dirs) == 2


def test_disk_sink_supports_events_without_model_confidence(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "aidetector.media.artifacts.generate_mp4",
        lambda *_args, **_kwargs: None,
    )
    item = make_observation(datetime.now(), {}, shape=(20, 30, 3))

    DiskSink(DiskConfig()).send(completed(make_event([item]), None))

    assert list((tmp_path / "detections" / "unclassified" / "unvalidated").iterdir())


def test_track_payload_and_detection_metadata_include_objects():
    event = make_event()
    tracks = tracks_payload(LiveObservation("0:0", event.best))
    detection = DetectionMetadata.from_event(event, True).as_dict()

    assert tracks["type"] == "tracks"
    assert tracks["source"] == "0:0"
    assert tracks["width"] == 120
    assert tracks["height"] == 80
    assert tracks["objects"] == [
        {
            "id": 0,
            "track_id": None,
            "crop": {
                "x1": 12,
                "y1": 12,
                "x2": 42,
                "y2": 52,
                "label": "cow",
                "confidence": 0.9,
                "track_id": None,
                "identity": None,
            },
        }
    ]
    assert detection["validated"] is True
    assert detection["confidence"] == 0.9
    assert detection["observations"] == 2
    assert detection["crops"] == [detection["crop"]]


def test_metadata_and_tracks_include_identity_results():
    item = make_observation(
        datetime(2026, 1, 1),
        {"cow": 0.9},
        objects=(
            DetectedObject(
                10,
                10,
                40,
                50,
                "cow",
                0.9,
                7,
                IdentityResult(
                    status="matched",
                    visual_identity_id="vid_123",
                    official_id="NL-123",
                    similarity=0.94,
                    margin=0.12,
                ),
            ),
        ),
    )
    event = make_event([item])

    tracks = tracks_payload(LiveObservation("0:0", item))
    metadata = DetectionMetadata.from_event(event, True).as_dict()

    identity = {
        "status": "matched",
        "visual_identity_id": "vid_123",
        "official_id": "NL-123",
        "similarity": 0.94,
        "margin": 0.12,
    }
    assert tracks["objects"][0]["crop"]["identity"] == identity
    assert metadata["identity_results"] == [identity]


def test_sse_server_is_shared_and_closed_by_last_sink(monkeypatch):
    class FakeServer:
        def __init__(self, port):
            self.port = port
            self.references = 0
            self.closed = False

        def hub(self, endpoint):
            return endpoint

        def close(self):
            self.closed = True

    monkeypatch.setattr("aidetector.adapters.sinks.sse.SSEServer", FakeServer)
    SSESink._servers.clear()
    port = 8765
    first = SSESink(SSEConfig(port=port, endpoint="/first"))
    second = SSESink(SSEConfig(port=port, endpoint="/second"))
    first.start()
    second.start()

    assert first.server is second.server
    server = first.server
    first.close()
    assert port in SSESink._servers
    assert server is not None and not server.closed
    second.close()
    assert port not in SSESink._servers
    assert server is not None and server.closed


def test_sse_hub_prioritizes_events_and_coalesces_tracks():
    hub = SSEHub("/events")
    client = hub.register()
    hub.publish("tracks", {"frame": 1})
    hub.publish("tracks", {"frame": 2})
    hub.publish("detection", {"event_id": "event"})

    event = client.read(0)
    track = client.read(0)

    assert event is not None and '"event_id":"event"' in event
    assert track is not None and '"frame":2' in track
    hub.close()
    assert client.read(0) is None


def test_webhook_sink_sends_no_body_for_none_data_type(monkeypatch):
    calls = []

    def fake_request(method, url, **kwargs):
        calls.append((method, url, kwargs))
        return type("Response", (), {"raise_for_status": lambda self: None})()

    monkeypatch.setattr(
        "aidetector.adapters.sinks.webhook.requests.request", fake_request
    )
    sink = WebhookSink(
        WebhookConfig(
            url="https://example.test/hook",
            method="GET",
            timeout=5,
            headers={"X-Test": "1"},
            data_type="none",
        )
    )

    sink.send(completed())

    assert calls == [
        (
            "GET",
            "https://example.test/hook",
            {"headers": {"X-Test": "1"}, "timeout": 5},
        )
    ]


def test_webhook_explicit_body_overrides_generated_payload(monkeypatch):
    calls = []

    def fake_request(method, url, **kwargs):
        calls.append((method, url, kwargs))
        return type("Response", (), {"raise_for_status": lambda self: None})()

    monkeypatch.setattr(
        "aidetector.adapters.sinks.webhook.requests.request", fake_request
    )
    sink = WebhookSink(
        WebhookConfig(
            url="https://example.test/hook",
            body="fixed-body",
            data_type="base64",
            include_image=True,
        )
    )

    sink.send(completed())

    request = calls[0][2]
    assert request["timeout"] == 30
    assert request["data"] == "fixed-body"
    assert "json" not in request
    assert "files" not in request


def test_webhook_encodes_base64_media(monkeypatch):
    calls = []

    def fake_request(method, url, **kwargs):
        calls.append(kwargs)
        return type("Response", (), {"raise_for_status": lambda self: None})()

    monkeypatch.setattr(
        "aidetector.adapters.sinks.webhook.requests.request", fake_request
    )
    monkeypatch.setattr(
        "aidetector.adapters.sinks.webhook.encode_media",
        lambda *_args, **_kwargs: {
            "image": EncodedFile("image.jpg", b"jpg", "image/jpeg")
        },
    )
    sink = WebhookSink(
        WebhookConfig(
            url="https://example.test/hook",
            data_type="base64",
            include_image=True,
        )
    )

    sink.send(completed())

    assert calls[0]["json"]["image"] == "anBn"
    assert calls[0]["json"]["event_id"]
    assert calls[0]["json"]["confidence"] == 0.9
    assert calls[0]["json"]["confidences"] == {"cow": 0.9}


def test_webhook_serializes_nested_metadata_for_multipart(monkeypatch):
    calls = []

    def fake_request(_method, _url, **kwargs):
        calls.append(kwargs)
        return type("Response", (), {"raise_for_status": lambda self: None})()

    monkeypatch.setattr(
        "aidetector.adapters.sinks.webhook.requests.request", fake_request
    )
    monkeypatch.setattr(
        "aidetector.adapters.sinks.webhook.encode_media",
        lambda *_args, **_kwargs: {},
    )

    WebhookSink(WebhookConfig(url="https://example.test/hook")).send(completed())

    assert json.loads(calls[0]["data"]["confidences"]) == {"cow": 0.9}
    assert json.loads(calls[0]["data"]["crops"])[0]["label"] == "cow"


def test_completed_event_has_one_validation_status():
    assert completed(validated=True).status == "approved"
    assert completed(validated=False).status == "rejected"
    assert completed(validated=None).status == "unvalidated"


def test_telegram_sink_respects_alert_every(monkeypatch):
    calls = []

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        return type("Response", (), {"raise_for_status": lambda self: None})()

    monkeypatch.setattr("aidetector.adapters.sinks.telegram.requests.post", fake_post)
    monkeypatch.setattr(
        "aidetector.adapters.sinks.telegram.encode_media",
        lambda *_args, **_kwargs: {
            "image": EncodedFile("image.jpg", b"jpg", "image/jpeg")
        },
    )
    sink = TelegramSink(
        ChatConfig(
            token="token",
            chat="chat-id",
            alert_every=2,
            include_image=True,
            include_video=False,
        )
    )

    sink.send(completed(validated=None))
    sink.send(completed(validated=None))

    assert calls[0][0].endswith("/sendPhoto")
    assert calls[0][1]["data"]["disable_notification"] is True
    assert calls[1][1]["data"]["disable_notification"] is False
    assert calls[1][1]["data"]["caption"].startswith("90%")


def test_telegram_sink_sends_text_when_media_is_unavailable(monkeypatch):
    calls = []

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        return type("Response", (), {"raise_for_status": lambda self: None})()

    monkeypatch.setattr("aidetector.adapters.sinks.telegram.requests.post", fake_post)
    monkeypatch.setattr(
        "aidetector.adapters.sinks.telegram.encode_media",
        lambda *_args, **_kwargs: {},
    )
    sink = TelegramSink(ChatConfig(token="token", chat="chat-id"))

    sink.send(completed())

    assert calls[0][0].endswith("/sendMessage")
    assert calls[0][1]["data"]["text"].startswith("90% approved")
