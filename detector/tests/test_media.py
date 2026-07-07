from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

import aidetector.media.video as video
from aidetector.media.video import (
    generate_mp4,
    get_crop,
    get_plot,
    shrink_image,
)
from aidetector.utils.config import Crop, DetectedObject, Detection, IdentityResult, ImageSet


def make_detection(crops: list[Crop]) -> Detection:
    image = np.zeros((100, 160, 3), dtype=np.uint8)
    return Detection(
        datetime(2026, 1, 1, 12, 0, 0),
        ImageSet(image, [DetectedObject(crop) for crop in crops]),
        {"cow": 0.9},
    )


def test_crop_region_is_union_of_all_crops():
    detection = make_detection(
        [
            Crop(10, 20, 30, 40, label="cow", confidence=0.9),
            Crop(50, 5, 80, 70, label="cow", confidence=0.8),
        ]
    )

    crop = detection.images.crop_region

    assert crop == Crop(10, 5, 80, 70)


def test_get_crop_uses_union_region_for_multiple_crops():
    detection = make_detection([Crop(10, 20, 30, 40), Crop(50, 5, 80, 70)])

    crop = get_crop(detection, aspect_ratio=None, padding=0, plot=False)

    assert crop is not None
    assert crop.shape == (65, 70, 3)


def test_get_plot_draws_multiple_crops():
    detection = make_detection([Crop(10, 20, 30, 40), Crop(50, 5, 80, 70)])

    plotted = get_plot(detection)

    assert plotted.shape == detection.images.jpg.shape
    assert np.any(plotted != detection.images.jpg)


def test_get_plot_draws_identity_above_detection_label():
    detection = make_detection([Crop(10, 60, 50, 90, label="cow", confidence=0.9)])
    detection.images.objects[0].identity = IdentityResult(
        identity="cow-main-0001",
        status="matched",
        similarity=0.8,
    )

    plotted = get_plot(detection)

    assert np.any(plotted[20:40, 10:80] != detection.images.jpg[20:40, 10:80])


def test_get_crop_can_plot_identity_label_for_carried_crops():
    detection = make_detection([Crop(10, 60, 50, 90, label="cow", confidence=0.9)])

    crop = get_crop(
        detection,
        aspect_ratio=None,
        padding=0,
        plot=True,
        identity_label="cow-main-0001 80%",
    )

    assert crop is not None
    assert crop.shape[0] > 30
    assert crop.shape[1] > 40
    assert np.any(crop[:20] != 0)


def test_generate_mp4_carries_event_identity_label(monkeypatch):
    identity_labels = []

    def fake_get_crop(*_args, identity_label=None, **_kwargs):
        identity_labels.append(identity_label)
        return np.zeros((10, 10, 3), dtype=np.uint8)

    class FakeProcess:
        def __init__(self, cmd, **_kwargs):
            self.cmd = cmd
            self.stdin = self
            self.returncode = 0

        def write(self, _data):
            pass

        def communicate(self):
            Path(self.cmd[-1]).write_bytes(b"mp4")
            return b"", b""

    first = make_detection([Crop(10, 20, 30, 40, label="mounting", confidence=0.9)])
    first.identities.append(
        IdentityResult(
            identity="cow-main-0001",
            status="matched",
            similarity=0.8,
        )
    )
    second = make_detection([Crop(12, 20, 32, 40, label="mounting", confidence=0.8)])
    second.date += timedelta(seconds=1)

    monkeypatch.setattr(video, "get_crop", fake_get_crop)
    monkeypatch.setattr(video, "get_ffmpeg_exe", lambda: "ffmpeg")
    monkeypatch.setattr(video.subprocess, "Popen", FakeProcess)

    assert video.generate_mp4([first, second]) == b"mp4"
    assert identity_labels == ["cow-main-0001 80%", "cow-main-0001 80%"]


def test_shrink_image_keeps_even_dimensions_and_does_not_upscale():
    image = np.zeros((101, 401, 3), dtype=np.uint8)

    shrunk = shrink_image(image, 200)
    unchanged = shrink_image(shrunk, 400)

    assert shrunk.shape[1] == 200
    assert shrunk.shape[0] % 2 == 0
    assert unchanged.shape == shrunk.shape


def test_generate_mp4_returns_none_without_detections():
    assert generate_mp4([]) is None
