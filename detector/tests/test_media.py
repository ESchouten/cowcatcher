from datetime import datetime

import numpy as np

from aidetector.media.video import generate_mp4, get_crop, get_plot, shrink_image
from aidetector.utils.config import Crop, Detection, ImageSet


def make_detection(crops: list[Crop]) -> Detection:
    image = np.zeros((100, 160, 3), dtype=np.uint8)
    return Detection(datetime(2026, 1, 1, 12, 0, 0), ImageSet(image, crops), {"cow": 0.9})


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


def test_shrink_image_keeps_even_dimensions_and_does_not_upscale():
    image = np.zeros((101, 401, 3), dtype=np.uint8)

    shrunk = shrink_image(image, 200)
    unchanged = shrink_image(shrunk, 400)

    assert shrunk.shape[1] == 200
    assert shrunk.shape[0] % 2 == 0
    assert unchanged.shape == shrunk.shape


def test_generate_mp4_returns_none_without_detections():
    assert generate_mp4([]) is None
