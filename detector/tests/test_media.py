import subprocess

import numpy as np

from aidetector.domain.detections import DetectedObject
from aidetector.media.rendering import (
    compress_jpg,
    get_crop,
    get_plot,
    shrink_image,
)
from aidetector.media.video import RenderedFrames, _encode_mp4, generate_mp4
from tests.factories import make_observation


def observation(objects: list[DetectedObject]):
    return make_observation(objects=tuple(objects), shape=(100, 160, 3))


def test_crop_region_is_union_of_all_crops():
    item = observation(
        [
            DetectedObject(10, 20, 30, 40, label="cow", confidence=0.9),
            DetectedObject(50, 5, 80, 70, label="cow", confidence=0.8),
        ]
    )

    crop = item.crop_region

    assert crop == DetectedObject(10, 5, 80, 70)


def test_get_crop_uses_union_region_for_multiple_crops():
    item = observation([DetectedObject(10, 20, 30, 40), DetectedObject(50, 5, 80, 70)])

    crop = get_crop(item, aspect_ratio=None, padding=0, plot=False)

    assert crop is not None
    assert crop.shape == (65, 70, 3)


def test_get_plot_draws_multiple_crops():
    item = observation([DetectedObject(10, 20, 30, 40), DetectedObject(50, 5, 80, 70)])

    plotted = get_plot(item)

    assert plotted.shape == item.frame.require_image().shape
    assert np.any(plotted != item.frame.require_image())


def test_shrink_image_keeps_even_dimensions_and_does_not_upscale():
    image = np.zeros((101, 401, 3), dtype=np.uint8)

    shrunk = shrink_image(image, 200)
    unchanged = shrink_image(shrunk, 400)

    assert shrunk.shape[1] == 200
    assert shrunk.shape[0] % 2 == 0
    assert unchanged.shape == shrunk.shape


def test_generate_mp4_returns_none_without_detections():
    assert generate_mp4([]) is None


def test_generate_mp4_handles_equal_frame_timestamps():
    observations = [observation([]), observation([])]

    video = generate_mp4(observations, width=160, crf=35, plot=False)

    assert video is not None
    assert video.startswith(b"\x00\x00")


def test_encode_mp4_kills_ffmpeg_after_timeout(monkeypatch):
    class Stdin:
        def write(self, _data):
            return None

    class Process:
        def __init__(self):
            self.stdin = Stdin()
            self.returncode = None
            self.killed = False

        def communicate(self, timeout=None):
            if timeout is not None:
                raise subprocess.TimeoutExpired("ffmpeg", timeout)
            return b"", b""

        def poll(self):
            return self.returncode

        def kill(self):
            self.killed = True
            self.returncode = -9

    process = Process()
    monkeypatch.setattr(
        "aidetector.media.video.subprocess.Popen",
        lambda *_args, **_kwargs: process,
    )
    item = observation([DetectedObject(10, 20, 30, 40)])

    assert _encode_mp4(RenderedFrames([item], False, False, 0), 1, 160, 28) is None
    assert process.killed


def test_compress_jpg_respects_maximum_for_compressible_image():
    image = np.zeros((800, 1200, 3), dtype=np.uint8)

    jpg = compress_jpg(image, 10_000)

    assert len(jpg) <= 10_000
