import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from aidetector.exporters.disk import DiskExporter
from aidetector.exporters.exporter import Exporter
from aidetector.exporters.sse import detection_payload, tracks_payload, write_tracks_log
from aidetector.exporters.telegram import TelegramExporter
from aidetector.exporters.webhook import WebhookExporter
from aidetector.utils.config import (
    ChatConfig,
    Crop,
    DetectedObject,
    Detection,
    DiskConfig,
    ExporterConfig,
    IdentityResult,
    ImageSet,
    WebhookConfig,
)


def make_detections() -> list[Detection]:
    start = datetime(2026, 1, 1, 12, 0, 0)
    image = np.zeros((80, 120, 3), dtype=np.uint8)
    return [
        Detection(
            start,
            ImageSet(
                image,
                [DetectedObject(Crop(10, 10, 40, 50, label="cow", confidence=0.7))],
            ),
            {"cow": 0.7},
        ),
        Detection(
            start + timedelta(seconds=2),
            ImageSet(
                image,
                [DetectedObject(Crop(12, 12, 42, 52, label="cow", confidence=0.9))],
            ),
            {"cow": 0.9},
        ),
    ]


class RecordingExporter(Exporter[ExporterConfig]):
    def __init__(self, config: ExporterConfig):
        super().__init__(config)
        self.calls = []

    def filtered_export(self, best_detection, detections, validated):
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
    monkeypatch.setattr("aidetector.exporters.disk.generate_mp4", lambda *_args, **_kwargs: b"mp4")
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


def test_webhook_exporter_sends_no_body_for_none_data_type(monkeypatch):
    calls = []

    def fake_request(method, url, **kwargs):
        calls.append((method, url, kwargs))
        return type("Response", (), {"status_code": 200, "text": ""})()

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


def test_sse_tracks_payload_contains_objects_and_identity():
    identity = IdentityResult(
        identity="cow-main-0001",
        status="matched",
        similarity=0.92,
    )
    detection = make_detections()[-1]
    detection.images.objects[0].track_id = 12
    detection.images.objects[0].identity = identity

    payload = tracks_payload("camera-1", detection)

    assert payload["type"] == "tracks"
    assert payload["source"] == "camera-1"
    assert payload["width"] == 120
    assert payload["height"] == 80
    assert payload["objects"] == [
        {
            "track_id": 12,
            "label": "cow",
            "confidence": 0.9,
            "crop": {"x1": 12, "y1": 12, "x2": 42, "y2": 52},
            "identity": {
                "identity": "cow-main-0001",
                "status": "matched",
                "similarity": 0.92,
            },
        }
    ]


def test_sse_tracks_log_writes_jsonl(tmp_path):
    log = tmp_path / "identity-sse.jsonl"
    payload = {
        "type": "tracks",
        "source": "camera-1",
        "objects": [{"track_id": 12, "identity": {"identity": "cow-main-0001"}}],
    }

    write_tracks_log(log, payload)

    assert json.loads(log.read_text()) == payload


def test_sse_detection_payload_contains_detection_summary():
    detections = make_detections()
    detections[-1].identities = [
        IdentityResult(
            identity="cow-main-0001",
            status="matched",
            similarity=0.92,
        )
    ]

    payload = detection_payload(detections[-1], detections, True)

    assert payload["type"] == "detection"
    assert payload["confidence"] == 0.9
    assert payload["validated"] is True
    assert payload["duration"] == 2
    assert payload["identity"]["identity"] == "cow-main-0001"
    assert payload["identities"][0]["similarity"] == 0.92


def test_webhook_explicit_body_overrides_generated_payload(monkeypatch):
    calls = []

    def fake_request(method, url, **kwargs):
        calls.append((method, url, kwargs))
        return type("Response", (), {"status_code": 200, "text": ""})()

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
    monkeypatch.setattr("aidetector.exporters.telegram.generate_mp4", lambda *_args, **_kwargs: None)
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

    first = exporter.get_payload(detections[-1], detections, None)
    second = exporter.get_payload(detections[-1], detections, None)

    assert first["chat_id"] == "chat-id"
    assert first["disable_notification"] is True
    assert second["disable_notification"] is False
    media = json.loads(second["media"])
    assert media[0]["caption"].startswith("90%")
