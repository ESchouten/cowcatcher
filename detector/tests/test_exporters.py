import json
from datetime import datetime, timedelta
from pathlib import Path
from queue import Queue

import numpy as np
import pytest

from aidetector.detection.models import Crop, Detection, ImageSet
from aidetector.detection.tracks import tracks_payload
from aidetector.exporters.disk import DiskExporter
from aidetector.exporters.exporter import Exporter
from aidetector.exporters.factory import build_exporters
from aidetector.exporters.media import EncodedFile
from aidetector.exporters.sse import SSEExporter, _SSEHub, detection_payload
from aidetector.exporters.telegram import TelegramExporter
from aidetector.exporters.webhook import WebhookExporter
from aidetector.utils.config import (
    ChatConfig,
    DiskConfig,
    ExporterConfig,
    ExportersConfig,
    SSEConfig,
    WebhookConfig,
)


def make_detections() -> list[Detection]:
    start = datetime(2026, 1, 1, 12, 0, 0)
    image = np.zeros((80, 120, 3), dtype=np.uint8)
    return [
        Detection(
            start,
            ImageSet(image, [Crop(10, 10, 40, 50, label="cow", confidence=0.7)]),
            {"cow": 0.7},
        ),
        Detection(
            start + timedelta(seconds=2),
            ImageSet(image, [Crop(12, 12, 42, 52, label="cow", confidence=0.9)]),
            {"cow": 0.9},
        ),
    ]


class RecordingExporter(Exporter[ExporterConfig]):
    def __init__(self, config: ExporterConfig):
        super().__init__(config)
        self.calls = []

    def _export(self, best_detection, detections, validated):
        self.calls.append((best_detection, detections, validated))


def test_exporter_filters_by_confidence_and_rejected_state():
    detections = make_detections()
    best = detections[-1]

    exporter = RecordingExporter(ExporterConfig(confidence=0.95))
    exporter.export(best, detections, True)
    assert exporter.calls == []

    exporter = RecordingExporter(ExporterConfig(confidence=0.5, export_rejected=False))
    exporter.export(best, detections, False)
    assert exporter.calls == []

    exporter = RecordingExporter(ExporterConfig(confidence=0.5, export_rejected=True))
    exporter.export(best, detections, False)
    assert len(exporter.calls) == 1


def test_disk_exporter_writes_detection_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "aidetector.exporters.disk.generate_mp4", lambda *_args, **_kwargs: b"mp4"
    )
    detections = make_detections()

    exporter = DiskExporter(DiskConfig(directory=Path("events")))
    exporter.export(detections[-1], detections, True)

    event_dirs = list((tmp_path / "detections" / "events" / "approved").iterdir())
    assert len(event_dirs) == 1
    event_dir = event_dirs[0]
    assert (event_dir / "best.jpg").exists()
    assert (event_dir / "clean.jpg").exists()
    assert (event_dir / "video.mp4").read_bytes() == b"mp4"

    metadata = json.loads((event_dir / "metadata.json").read_text())
    assert metadata["validated"] is True
    assert metadata["confidence"] == 0.9
    assert metadata["detections"] == 2
    assert metadata["crop"] == {"x1": 12, "y1": 12, "x2": 42, "y2": 52}


def test_disk_exporter_supports_events_without_model_confidence(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "aidetector.exporters.disk.generate_mp4",
        lambda *_args, **_kwargs: None,
    )
    item = Detection(
        datetime.now(),
        ImageSet(np.zeros((20, 30, 3), dtype=np.uint8)),
        {},
    )

    DiskExporter(DiskConfig()).export(item, [item], None)

    assert list((tmp_path / "detections" / "unclassified" / "unvalidated").iterdir())


def test_sse_payloads_include_live_crops_and_detection_summary():
    detections = make_detections()

    tracks = tracks_payload("0:0", detections[-1])
    detection = detection_payload(detections[-1], detections, True)

    assert tracks["type"] == "tracks"
    assert tracks["source"] == "0:0"
    assert tracks["width"] == 120
    assert tracks["height"] == 80
    assert tracks["objects"] == [
        {
            "id": 0,
            "track_id": None,
            "label": "cow",
            "confidence": 0.9,
            "crop": {
                "x1": 12,
                "y1": 12,
                "x2": 42,
                "y2": 52,
                "label": "cow",
                "confidence": 0.9,
            },
        }
    ]
    assert detection["type"] == "detection"
    assert detection["validated"] is True
    assert detection["confidence"] == 0.9
    assert detection["detections"] == 2
    assert detection["crop"] == {
        "x1": 12,
        "y1": 12,
        "x2": 42,
        "y2": 52,
        "label": "cow",
        "confidence": 0.9,
    }


def test_sse_server_is_shared_and_closed_by_last_exporter(monkeypatch):
    class FakeServer:
        def __init__(self, port):
            self.port = port
            self.references = 0
            self.closed = False

        def hub(self, endpoint):
            return endpoint

        def close(self):
            self.closed = True

    monkeypatch.setattr("aidetector.exporters.sse._SSEServer", FakeServer)
    SSEExporter._servers.clear()
    port = 8765
    first = SSEExporter(SSEConfig(port=port, endpoint="/first"))
    second = SSEExporter(SSEConfig(port=port, endpoint="/second"))

    assert first.server is second.server
    first.close()
    assert port in SSEExporter._servers
    assert not first.server.closed
    second.close()
    assert port not in SSEExporter._servers
    assert first.server.closed


def test_exporter_factory_closes_sse_exporters_after_partial_failure(monkeypatch):
    created = []

    class FailingSSEExporter:
        def __init__(self, config):
            if created:
                raise OSError("port unavailable")
            self.closed = False
            created.append(self)

        def close(self):
            self.closed = True

    monkeypatch.setattr("aidetector.exporters.factory.SSEExporter", FailingSSEExporter)

    with pytest.raises(OSError, match="port unavailable"):
        build_exporters(
            ExportersConfig(
                sse=[SSEConfig(endpoint="/one"), SSEConfig(endpoint="/two")]
            ),
            detector_index=0,
        )

    assert created[0].closed is True


def test_sse_hub_close_discards_pending_messages():
    hub = _SSEHub("/events")
    client: Queue[str | None] = hub.register()
    hub.publish("tracks", {"frame": 1})
    hub.publish("tracks", {"frame": 2})

    hub.close()

    assert client.get_nowait() is None
    assert client.empty()


def test_webhook_exporter_sends_no_body_for_none_data_type(monkeypatch):
    calls = []

    def fake_request(method, url, **kwargs):
        calls.append((method, url, kwargs))
        return type("Response", (), {"raise_for_status": lambda self: None})()

    monkeypatch.setattr("aidetector.exporters.webhook.requests.request", fake_request)
    detections = make_detections()
    exporter = WebhookExporter(
        WebhookConfig(
            url="https://example.test/hook",
            method="GET",
            timeout=5,
            headers={"X-Test": "1"},
            data_type="none",
        )
    )

    exporter.export(detections[-1], detections, True)

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

    monkeypatch.setattr("aidetector.exporters.webhook.requests.request", fake_request)
    detections = make_detections()
    exporter = WebhookExporter(
        WebhookConfig(
            url="https://example.test/hook",
            body="fixed-body",
            data_type="base64",
            include_image=True,
        )
    )

    exporter.export(detections[-1], detections, True)

    request = calls[0][2]
    assert request["data"] == "fixed-body"
    assert "json" not in request
    assert "files" not in request


def test_telegram_exporter_respects_alert_every(monkeypatch):
    calls = []

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        return type("Response", (), {"raise_for_status": lambda self: None})()

    monkeypatch.setattr("aidetector.exporters.telegram.requests.post", fake_post)
    monkeypatch.setattr(
        "aidetector.exporters.telegram.encode_media",
        lambda *_args, **_kwargs: {
            "image": EncodedFile("image.jpg", b"jpg", "image/jpeg")
        },
    )
    detections = make_detections()
    exporter = TelegramExporter(
        ChatConfig(
            token="token",
            chat="chat-id",
            alert_every=2,
            include_image=True,
            include_video=False,
        )
    )

    exporter.export(detections[-1], detections, None)
    exporter.export(detections[-1], detections, None)

    assert calls[0][0].endswith("/sendPhoto")
    assert calls[0][1]["data"]["chat_id"] == "chat-id"
    assert calls[0][1]["data"]["disable_notification"] is True
    assert calls[1][1]["data"]["disable_notification"] is False
    assert calls[1][1]["data"]["caption"].startswith("90%")


def test_telegram_exporter_sends_text_when_media_is_unavailable(monkeypatch):
    calls = []

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        return type("Response", (), {"raise_for_status": lambda self: None})()

    monkeypatch.setattr("aidetector.exporters.telegram.requests.post", fake_post)
    monkeypatch.setattr(
        "aidetector.exporters.telegram.encode_media",
        lambda *_args, **_kwargs: {},
    )
    detections = make_detections()
    exporter = TelegramExporter(ChatConfig(token="token", chat="chat-id"))

    exporter.export(detections[-1], detections, True)

    assert calls[0][0].endswith("/sendMessage")
    assert calls[0][1]["data"]["text"].startswith("90% approved")
