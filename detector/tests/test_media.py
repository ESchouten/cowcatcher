from datetime import datetime

import numpy as np

from aidetector.media.video import (
    _crop_identity_label,
    generate_mp4,
    get_crop,
    get_plot,
    shrink_image,
)
from aidetector.utils.config import Crop, Detection, IdentityResult, ImageSet


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


def test_get_plot_draws_identity_above_detection_label():
    detection = make_detection([Crop(10, 60, 50, 90, label="cow", confidence=0.9)])
    detection.identities = [
        IdentityResult(
            provider="cow-main",
            identity_id="cow-main-0001",
            name=None,
            status="matched",
            similarity=0.8,
        )
    ]

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


def test_crop_identity_label_uses_fallback_for_first_crop_only():
    identity_label = "cow-main-0001 80%, cow-main-0002 70%"

    assert _crop_identity_label(0, identity_label) == identity_label
    assert _crop_identity_label(1, identity_label) is None


def test_shrink_image_keeps_even_dimensions_and_does_not_upscale():
    image = np.zeros((101, 401, 3), dtype=np.uint8)

    shrunk = shrink_image(image, 200)
    unchanged = shrink_image(shrunk, 400)

    assert shrunk.shape[1] == 200
    assert shrunk.shape[0] % 2 == 0
    assert unchanged.shape == shrunk.shape


def test_generate_mp4_returns_none_without_detections():
    assert generate_mp4([]) is None
