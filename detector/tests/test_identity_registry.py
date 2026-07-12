from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np

from aidetector.detection.identity_registry import IdentityRegistry
from aidetector.utils.config import Crop, Detection, IdentityResult, ImageSet


def detection(width: int, height: int, crops: list[Crop]) -> Detection:
    return Detection(
        datetime(2026, 1, 1),
        ImageSet(np.zeros((height, width, 3), dtype=np.uint8), crops),
        {},
    )


def test_registry_enriches_multiple_detectors_using_normalized_location():
    registry = IdentityRegistry()
    registry.publish(
        "camera",
        "identity-detector",
        detection(
            100,
            100,
            [
                Crop(
                    10,
                    10,
                    30,
                    30,
                    track_id=1,
                    identities=[IdentityResult("001", 0.95)],
                ),
                Crop(
                    60,
                    10,
                    80,
                    30,
                    track_id=2,
                    identities=[IdentityResult("002", 0.9)],
                ),
            ],
        ),
    )
    event = detection(200, 200, [Crop(0, 0, 80, 80, label="mounting")])

    registry.enrich("camera", event)

    assert event.images.crops[0].identities == [IdentityResult("001", 0.95)]


def test_registry_keeps_tracker_namespaces_separate_and_expires_observations():
    now = 0.0
    registry = IdentityRegistry(ttl=5, clock=lambda: now)
    registry.publish(
        "camera",
        "detector-1",
        detection(
            100,
            100,
            [
                Crop(
                    10,
                    10,
                    30,
                    30,
                    track_id=1,
                    identities=[IdentityResult("001", 0.95)],
                )
            ],
        ),
    )
    registry.publish(
        "camera",
        "detector-2",
        detection(
            100,
            100,
            [
                Crop(
                    60,
                    10,
                    80,
                    30,
                    track_id=1,
                    identities=[IdentityResult("002", 0.9)],
                )
            ],
        ),
    )
    visible = detection(100, 100, [Crop(0, 0, 100, 50)])

    registry.enrich("camera", visible)
    assert visible.images.crops[0].identities == [
        IdentityResult("001", 0.95),
        IdentityResult("002", 0.9),
    ]

    now = 6
    expired = detection(100, 100, [Crop(0, 0, 100, 50)])
    registry.enrich("camera", expired)
    assert expired.images.crops[0].identities == []


def test_registry_accepts_concurrent_detector_updates():
    registry = IdentityRegistry()

    def publish(index: int) -> None:
        registry.publish(
            "camera",
            f"detector-{index}",
            detection(
                100,
                100,
                [
                    Crop(
                        10,
                        10,
                        30,
                        30,
                        track_id=1,
                        identities=[IdentityResult(f"cow-{index}", 0.9)],
                    )
                ],
            ),
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(publish, range(10)))

    event = detection(100, 100, [Crop(0, 0, 50, 50)])
    registry.enrich("camera", event)
    assert {identity.identity for identity in event.identities} == {
        f"cow-{index}" for index in range(10)
    }
