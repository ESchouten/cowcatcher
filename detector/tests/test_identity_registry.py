from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np

from aidetector.adapters.models.identity import IdentityEnricher
from aidetector.domain.detections import (
    DetectedObject,
    IdentityResult,
    Observation,
    unique_identities,
)
from aidetector.domain.frames import Frame
from aidetector.pipeline.identity import IdentityRegistry
from aidetector.pipeline.ports import ModelBatchResult


def observation(
    width: int,
    height: int,
    objects: list[DetectedObject],
) -> Observation:
    return Observation(
        Frame(datetime(2026, 1, 1), np.zeros((height, width, 3), dtype=np.uint8)),
        tuple(objects),
    )


def test_registry_enriches_multiple_detectors_using_normalized_location():
    registry = IdentityRegistry()
    registry.publish(
        "camera",
        "identity-detector",
        observation(
            100,
            100,
            [
                DetectedObject(
                    10,
                    10,
                    30,
                    30,
                    track_id=1,
                    identities=(IdentityResult("001", 0.95),),
                ),
                DetectedObject(
                    60,
                    10,
                    80,
                    30,
                    track_id=2,
                    identities=(IdentityResult("002", 0.9),),
                ),
            ],
        ),
    )
    event = observation(
        200,
        200,
        [DetectedObject(0, 0, 80, 80, label="mounting")],
    )

    enriched = registry.enrich("camera", event)

    assert enriched.objects[0].identities == (IdentityResult("001", 0.95),)


def test_registry_keeps_tracker_namespaces_separate_and_expires_observations():
    now = 0.0
    registry = IdentityRegistry(ttl=5, clock=lambda: now)
    for producer, x1, identity in (
        ("detector-1", 10, "001"),
        ("detector-2", 60, "002"),
    ):
        registry.publish(
            "camera",
            producer,
            observation(
                100,
                100,
                [
                    DetectedObject(
                        x1,
                        10,
                        x1 + 20,
                        30,
                        track_id=1,
                        identities=(IdentityResult(identity, 0.9),),
                    )
                ],
            ),
        )

    visible = registry.enrich(
        "camera",
        observation(100, 100, [DetectedObject(0, 0, 100, 50)]),
    )
    assert visible.objects[0].identities == (
        IdentityResult("001", 0.9),
        IdentityResult("002", 0.9),
    )

    now = 6
    expired = registry.enrich(
        "camera",
        observation(100, 100, [DetectedObject(0, 0, 100, 50)]),
    )
    assert expired.objects[0].identities == ()


def test_registry_accepts_concurrent_detector_updates():
    registry = IdentityRegistry()

    def publish(index: int) -> None:
        registry.publish(
            "camera",
            f"detector-{index}",
            observation(
                100,
                100,
                [
                    DetectedObject(
                        10,
                        10,
                        30,
                        30,
                        track_id=1,
                        identities=(IdentityResult(f"cow-{index}", 0.9),),
                    )
                ],
            ),
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(publish, range(10)))

    event = registry.enrich(
        "camera",
        observation(100, 100, [DetectedObject(0, 0, 50, 50)]),
    )
    assert {identity.identity for identity in unique_identities(event.objects)} == {
        f"cow-{index}" for index in range(10)
    }


def test_identity_enricher_reuses_primary_tracking_result():
    frame = Frame(
        datetime(2026, 1, 1),
        np.zeros((100, 100, 3), dtype=np.uint8),
    )
    tracked = ModelBatchResult("camera", object(), [frame])

    class Runner:
        def track_sources(self, _batch):
            return [tracked]

        def observations_from_result(self, _result, _frames):
            return [
                Observation(
                    frame,
                    (DetectedObject(0, 0, 50, 50, label="mounting"),),
                )
            ]

    class Pipeline:
        reuses_primary_yolo = True

        def candidates_from_primary(self, _source, _result, _image):
            return [object()]

        def live_observation(self, _frame, _candidates):
            return Observation(
                frame,
                (
                    DetectedObject(
                        10,
                        10,
                        30,
                        30,
                        track_id=7,
                        identities=(IdentityResult("cow-0001", 0.93),),
                    ),
                ),
            )

        def close(self):
            pass

    runner = Runner()
    enricher = IdentityEnricher(
        IdentityRegistry(),
        "detector-0",
        Pipeline(),
        runner,
    )

    result = enricher.process({"camera": [frame]})

    assert result.model_results == (tracked,)
    assert result.observations[0][1].objects[0].identities == (
        IdentityResult("cow-0001", 0.93),
    )
