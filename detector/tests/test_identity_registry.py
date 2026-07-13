from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np

from aidetector.domain.detections import (
    DetectedObject,
    Observation,
    unique_identities,
)
from aidetector.domain.frames import Frame
from aidetector.domain.identity import IdentityCandidate, IdentityResult
from aidetector.pipeline.identity import IdentityEnricher, IdentityRegistry
from aidetector.pipeline.identity_provider import IdentityBatch
from aidetector.pipeline.identity_provider import ModelIdentityCandidateSource
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
                        identities=(IdentityResult(f"identity-{index}", 0.9),),
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
        f"identity-{index}" for index in range(10)
    }


def test_identity_enricher_reuses_primary_tracking_result():
    frame = Frame(
        datetime(2026, 1, 1),
        np.zeros((100, 100, 3), dtype=np.uint8),
    )
    tracked = ModelBatchResult("camera", object(), [frame])

    class Provider:
        def start(self):
            pass

        def process(self, _batch):
            detection = DetectedObject(
                10,
                10,
                30,
                30,
                track_id=7,
                identities=(IdentityResult("identity-0001", 0.93),),
            )
            return [
                IdentityBatch(
                    "camera",
                    [frame],
                    (IdentityCandidate(detection, frame.require_image()),),
                    tracked,
                    Observation(
                        frame,
                        (DetectedObject(0, 0, 50, 50, label="mounting"),),
                    ),
                )
            ]

        def close(self):
            pass

    enricher = IdentityEnricher(
        IdentityRegistry(),
        "detector-0",
        Provider(),
    )

    result = enricher.process({"camera": [frame]})

    assert result.model_results == (tracked,)
    assert result.observations[0][1].objects[0].identities == (
        IdentityResult("identity-0001", 0.93),
    )


def test_model_identity_candidate_source_can_reuse_detection_result():
    frame = Frame(
        datetime(2026, 1, 1),
        np.zeros((100, 100, 3), dtype=np.uint8),
    )
    raw_result = object()
    tracked = ModelBatchResult("camera", raw_result, [frame])
    cow = IdentityCandidate(
        DetectedObject(10, 10, 30, 30, label="cow", track_id=7),
        frame.require_image()[10:30, 10:30],
    )

    class Runner:
        def track_sources(self, _batch):
            return [tracked]

        def observations_from_result(self, result, frames):
            assert result is raw_result
            return [
                Observation(
                    frames[-1],
                    (DetectedObject(0, 0, 50, 50, label="mounting"),),
                )
            ]

    class Localizer:
        def candidates(self, result, image):
            assert result is raw_result
            assert image is frame.require_image()
            return [cow]

    source = ModelIdentityCandidateSource(
        Runner(),
        Localizer(),
        reuse_for_detection=True,
    )

    result = source.batches({"camera": [frame]})[0]

    assert result.candidates == (cow,)
    assert result.model_result is tracked
    assert result.detection_observation is not None
    assert result.detection_observation.objects[0].label == "mounting"
