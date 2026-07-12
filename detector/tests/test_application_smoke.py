import json
from datetime import datetime

import cv2
import numpy as np

from aidetector.application import Application
from aidetector.domain.detections import DetectedObject, Observation
from aidetector.utils.config import (
    Config,
    DetectionConfig,
    DetectorConfig,
    DiskConfig,
    ExportersConfig,
    YoloConfig,
)


class FakeYoloRunner:
    def __init__(self, _config, _sources, _model):
        pass

    def detect(self, frames):
        return frames

    def track_sources(self, _batch):
        raise AssertionError("Tracking is not enabled in this smoke test")

    def observations_from_result(self, result, frames):
        height, width = result.shape[:2]
        return [
            Observation(
                frames[-1],
                (
                    DetectedObject(
                        width // 4,
                        height // 4,
                        width * 3 // 4,
                        height * 3 // 4,
                        label="cow",
                        confidence=0.9,
                    ),
                ),
                {"cow": 0.9},
            )
        ]


def test_application_processes_source_and_writes_detection(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    image_path = tmp_path / "frame.jpg"
    image = np.full((64, 96, 3), 127, dtype=np.uint8)
    assert cv2.imwrite(str(image_path), image)

    monkeypatch.setattr("aidetector.application.YoloRunner", FakeYoloRunner)
    monkeypatch.setattr(
        "aidetector.application.build_yolo_model",
        lambda *_args: object(),
    )
    monkeypatch.setattr(
        "aidetector.media.artifacts.generate_mp4",
        lambda *_args, **_kwargs: None,
    )
    config = Config(
        detectors=[
            DetectorConfig(
                detection=DetectionConfig(source=str(image_path)),
                yolo=YoloConfig(
                    model="fake.pt",
                    confidence=0.5,
                    frames_min=1,
                    time_max=0,
                ),
                exporters=ExportersConfig(disk=DiskConfig(directory="smoke")),
            )
        ]
    )
    application = Application.from_config(config)

    application.start()
    try:
        application.wait()
    finally:
        application.stop()

    event_directories = list(
        (tmp_path / "detections" / "smoke" / "unvalidated").iterdir()
    )
    assert len(event_directories) == 1
    event_directory = event_directories[0]
    assert (event_directory / "best.jpg").exists()
    assert (event_directory / "clean.jpg").exists()

    metadata = json.loads((event_directory / "metadata.json").read_text())
    assert metadata["validated"] is None
    assert metadata["confidence"] == 0.9
    assert metadata["confidences"] == {"cow": 0.9}
    assert datetime.fromisoformat(metadata["start"])
