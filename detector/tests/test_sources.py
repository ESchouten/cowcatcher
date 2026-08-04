from threading import Event
from time import monotonic

import numpy as np
import pytest

from aidetector.services.healthcheck import Healthcheck
from aidetector.adapters.sources.collector import FrameCollector
from aidetector.adapters.sources.source import SourceProvider
from aidetector.utils.config import DetectionConfig, HealthcheckConfig


def test_frame_collector_enforces_exact_retention_and_takes_snapshot():
    collector = FrameCollector(width=100, retention=2)
    for value in range(3):
        collector.add(
            "camera",
            np.full((20, 40, 3), value, dtype=np.uint8),
        )

    snapshot = collector.take()

    assert len(snapshot["camera"]) == 2
    assert snapshot["camera"][0].require_image()[0, 0, 0] == 1
    assert collector.frames == {}


def test_source_provider_rejects_mixed_files_and_streams():
    with pytest.raises(ValueError, match="cannot mix"):
        SourceProvider(
            DetectionConfig(source=["recording.mp4", "rtsp://camera/stream"])
        )


def test_healthcheck_stop_interrupts_long_interval(monkeypatch):
    requested = Event()

    def request(*_args, **_kwargs):
        requested.set()
        return type("Response", (), {"status_code": 200})()

    monkeypatch.setattr("aidetector.services.healthcheck.requests.request", request)
    healthcheck = Healthcheck(
        HealthcheckConfig(
            url="https://example.test/health",
            interval=60,
            timeout=1,
        )
    )
    healthcheck.start()
    assert requested.wait(1)

    started = monotonic()
    healthcheck.stop()

    assert monotonic() - started < 1
