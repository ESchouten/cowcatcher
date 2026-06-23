import tempfile
import unittest
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
from aidetector.detection.detector import Detector
from aidetector.detection.yolo import apply_mask, objects_from_result
from aidetector.exporters.telegram import TelegramExporter
from aidetector.exporters.webhook import WebhookExporter
from aidetector.identity.store import SQLiteIdentityStore
from aidetector.identity.wildlife_tools import (
    WildlifeToolsIdentityProvider,
    _model_signature,
)
from aidetector.utils.config import (
    Crop,
    Detection,
    DetectorIdentityConfig,
    IdentityProviderConfig,
    IdentityResult,
    ImageSet,
    YoloConfig,
)


class IdentityStoreTests(unittest.TestCase):
    def test_identity_store_reloads_identities(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "identities.sqlite"
            store = SQLiteIdentityStore(database, "cow-main")
            identity_id = store.create_identity(np.array([1, 0], dtype=np.float32))
            store.close()

            reloaded = SQLiteIdentityStore(database, "cow-main")
            try:
                result = reloaded.identify(
                    np.array([1, 0], dtype=np.float32),
                    source="source-1",
                    match_threshold=0.75,
                    candidate_threshold=0.75,
                    create_after=3,
                )
            finally:
                reloaded.close()

        self.assertEqual(identity_id, "cow-main-0001")
        self.assertEqual(result.identity_id, identity_id)
        self.assertEqual(result.status, "matched")

    def test_open_set_flow_creates_identity_after_repeated_sightings(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "identities.sqlite"
            store = SQLiteIdentityStore(
                database, "cow-main"
            )
            try:
                results = [
                    store.identify(
                        np.array([1, 0], dtype=np.float32),
                        source="source-1",
                        match_threshold=0.75,
                        candidate_threshold=0.75,
                        create_after=3,
                    )
                    for _ in range(3)
                ]
                sample_count = store.connection.execute(
                    "SELECT COUNT(*) FROM samples WHERE identity_id = ?",
                    ("cow-main-0001",),
                ).fetchone()[0]
            finally:
                store.close()

        self.assertEqual([result.status for result in results], [
            "unknown",
            "unknown",
            "created",
        ])
        self.assertEqual(results[-1].identity_id, "cow-main-0001")
        self.assertEqual(sample_count, 3)

    def test_known_identity_above_threshold_matches(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteIdentityStore(
                Path(directory) / "identities.sqlite", "cow-main"
            )
            try:
                identity_id = store.create_identity(
                    np.array([1, 0], dtype=np.float32)
                )
                result = store.identify(
                    np.array([0.9, 0.1], dtype=np.float32),
                    source="source-1",
                    match_threshold=0.75,
                    candidate_threshold=0.75,
                    create_after=3,
                )
            finally:
                store.close()

        self.assertEqual(result.status, "matched")
        self.assertEqual(result.identity_id, identity_id)

    def test_store_rejects_model_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "identities.sqlite"
            store = SQLiteIdentityStore(database, "cow-main", model="model-a")
            store.close()

            with self.assertRaises(ValueError):
                SQLiteIdentityStore(database, "cow-main", model="model-b")

    def test_store_rejects_embedding_dimension_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteIdentityStore(
                Path(directory) / "identities.sqlite", "cow-main"
            )
            try:
                store.identify(
                    np.array([1, 0], dtype=np.float32),
                    source="source-1",
                    match_threshold=0.75,
                    candidate_threshold=0.75,
                    create_after=3,
                )

                with self.assertRaises(ValueError):
                    store.identify(
                        np.array([1, 0, 0], dtype=np.float32),
                        source="source-1",
                        match_threshold=0.75,
                        candidate_threshold=0.75,
                        create_after=3,
                    )
            finally:
                store.close()


class DetectorIdentityTests(unittest.TestCase):
    def test_yolo_result_keeps_multiple_crops_per_frame(self):
        detector = Detector.__new__(Detector)
        detector.yolo_config = YoloConfig(model="cow.pt", confidence={"cow": 0.5})
        detector.yolo_class_confidences = {19: ("cow", 0.5)}
        processed = []
        detector._process = lambda source, detections: processed.append(detections)
        result = _SegmentationResult(
            names={19: "cow"},
            boxes=[
                _Box(19, 0.7, [1, 1, 4, 4]),
                _Box(19, 0.9, [5, 5, 9, 9]),
            ],
            masks=[],
            orig_shape=(10, 10),
        )

        detector._handle_yolo_result(
            "source-1",
            result,
            [(datetime(2026, 1, 1, 12, 0, 0), np.zeros((10, 10, 3), dtype=np.uint8))],
        )

        detection = processed[0][0]
        self.assertEqual(len(detection.images.crops), 2)
        self.assertEqual(detection.images.crop.x1, 5)
        self.assertEqual(detection.confidence, {"cow": 0.9})

    def test_detector_identity_label_filtering(self):
        detector = Detector.__new__(Detector)
        detector.identity_config = _detector_identity_config(labels=["cow"])

        self.assertTrue(detector._identity_matches_label(_detection(label="cow")))
        self.assertFalse(detector._identity_matches_label(_detection(label="deer")))
        self.assertTrue(
            detector._identity_matches_label(
                _detection(label=None, confidence={"cow": 0.9})
            )
        )
        self.assertFalse(
            detector._identity_matches_label(
                _detection(label="deer", confidence={"cow": 0.9})
            )
        )

    def test_identity_failure_does_not_block_export(self):
        detector = _detector_for_export(validated=True)
        exporter = detector.exporters[0]

        detector._export("source-1")
        detector.export_executor.shutdown(wait=True)

        self.assertTrue(exporter.called)

    def test_rejected_detection_skips_identity_lookup(self):
        detector = _detector_for_export(validated=False)
        identity_service = detector.identity_service

        detector._export("source-1")
        detector.export_executor.shutdown(wait=True)

        self.assertEqual(identity_service.calls, 0)

    def test_detector_identity_multiple_flag_collects_provider_results(self):
        detector = Detector.__new__(Detector)
        detector.identity_config = _detector_identity_config(
            multiple=True,
        )
        detector.identity_service = _ListIdentityService()
        detection = _detection()

        detector._identify("source-1", detection)

        self.assertTrue(detector.identity_service.multiple)
        self.assertEqual(len(detection.identities), 2)
        self.assertEqual(detection.identity.identity_id, "cow-main-0001")

    def test_detector_identity_samples_event_detections(self):
        detector = Detector.__new__(Detector)
        detector.identity_config = _detector_identity_config(
            samples=3,
        )
        detector.identity_service = _RecordingIdentityService()
        detections = [
            _detection(date=datetime(2026, 1, 1, 12, 0, 0), confidence={"cow": 0.6}),
            _detection(date=datetime(2026, 1, 1, 12, 0, 1), confidence={"cow": 0.7}),
            _detection(date=datetime(2026, 1, 1, 12, 0, 2), confidence={"cow": 0.8}),
            _detection(date=datetime(2026, 1, 1, 12, 0, 3), confidence={"cow": 0.9}),
        ]
        best_detection = detections[2]

        detector._identify("source-1", best_detection, detections)

        identity_input = detector.identity_service.detection
        self.assertIsInstance(identity_input, list)
        self.assertEqual(len(identity_input), 3)
        self.assertTrue(any(sample.date == best_detection.date for sample in identity_input))
        self.assertEqual(best_detection.identity.identity_id, "cow-main-0001")

    def test_detector_identity_samples_keep_frame_order_after_adding_best(self):
        detector = Detector.__new__(Detector)
        detector.identity_config = _detector_identity_config(
            samples=3,
        )
        detections = [
            _detection(date=datetime(2026, 1, 1, 12, 0, index))
            for index in range(5)
        ]

        samples = detector._identity_sample_detections(detections, detections[1])

        self.assertEqual(
            [sample.date.second for sample in samples],
            [0, 1, 2],
        )


class WildlifeToolsIdentityTests(unittest.TestCase):
    def test_objects_from_result_picks_accepted_labels(self):
        result = _SegmentationResult(
            names={0: "person", 19: "cow"},
            boxes=[
                _Box(19, 0.8, [0, 0, 1, 1]),
                _Box(0, 0.9, [0, 0, 2, 2]),
                _Box(19, 0.9, [0, 0, 2, 2]),
            ],
            masks=[
                [[1, 0], [0, 0]],
                [[1, 1], [1, 1]],
                [[1, 1], [0, 1]],
            ],
        )

        objects = objects_from_result(result, (2, 2), labels=["cow"])

        self.assertEqual(len(objects), 2)
        mask = max(objects, key=lambda obj: obj.area).mask
        np.testing.assert_array_equal(
            mask,
            np.array([[True, True], [False, True]]),
        )

    def test_apply_mask_neutralizes_background(self):
        image = np.arange(12, dtype=np.uint8).reshape((2, 2, 3))
        mask = np.array([[True, False], [False, True]])

        masked = apply_mask(image, mask, "gray")

        np.testing.assert_array_equal(masked[0, 0], image[0, 0])
        np.testing.assert_array_equal(masked[1, 1], image[1, 1])
        np.testing.assert_array_equal(masked[0, 1], np.array([127, 127, 127]))
        np.testing.assert_array_equal(masked[1, 0], np.array([127, 127, 127]))

    def test_provider_skips_identity_when_segment_model_finds_no_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            provider = WildlifeToolsIdentityProvider(
                IdentityProviderConfig(
                    id="cow-main",
                    database=Path(directory) / "identities.sqlite",
                    segment_model="seg.pt",
                )
            )
            provider.segmenter = _Segmenter([])
            try:
                result = provider.identify(_detection(), "source-1")
            finally:
                provider.close()

        self.assertIsNone(result)

    def test_provider_skips_segment_identity_without_crop(self):
        with tempfile.TemporaryDirectory() as directory:
            provider = WildlifeToolsIdentityProvider(
                IdentityProviderConfig(
                    id="cow-main",
                    database=Path(directory) / "identities.sqlite",
                    segment_model="seg.pt",
                )
            )
            provider.segmenter = _Segmenter([
                _SegmentationResult(
                    names={19: "cow"},
                    boxes=[_Box(19, 0.9, [0, 0, 2, 2])],
                    masks=[[[1, 1], [1, 1]]],
                )
            ])
            try:
                result = provider.identify(_detection(label=None), "source-1")
            finally:
                provider.close()

        self.assertIsNone(result)

    def test_provider_crops_to_segmented_object(self):
        image = np.arange(10 * 10 * 3, dtype=np.uint8).reshape((10, 10, 3))
        mask = np.zeros((10, 10), dtype=bool)
        mask[2:8, 1:6] = True

        with tempfile.TemporaryDirectory() as directory:
            provider = WildlifeToolsIdentityProvider(
                IdentityProviderConfig(
                    id="cow-main",
                    database=Path(directory) / "identities.sqlite",
                    segment_model="seg.pt",
                    segment_confidence=0.5,
                )
            )
            provider.segmenter = _Segmenter([
                _SegmentationResult(
                    names={19: "cow"},
                    boxes=[_Box(19, 0.9, [1, 2, 6, 8])],
                    masks=[mask],
                    orig_shape=(10, 10),
                )
            ])
            try:
                segments = provider._mask_identity_images(
                    image,
                    "source-1",
                    "mounting",
                )
            finally:
                provider.close()

        self.assertEqual(len(segments), 1)
        segmented = segments[0]
        self.assertEqual(segmented.shape, (6, 5, 3))
        np.testing.assert_array_equal(segmented[0, 0], image[2, 1])

    def test_provider_returns_multiple_segment_identities_when_requested(self):
        first_mask = np.zeros((7, 7), dtype=bool)
        first_mask[0:4, 0:4] = True
        second_mask = np.zeros((7, 7), dtype=bool)
        second_mask[4:7, 4:7] = True

        with tempfile.TemporaryDirectory() as directory:
            provider = WildlifeToolsIdentityProvider(
                IdentityProviderConfig(
                    id="cow-main",
                    database=Path(directory) / "identities.sqlite",
                    segment_model="seg.pt",
                    segment_confidence=0.5,
                    create_after=1,
                    crop_padding=0,
                )
            )
            provider.segmenter = _Segmenter([
                _SegmentationResult(
                    names={19: "cow"},
                    boxes=[
                        _Box(19, 0.9, [0, 0, 4, 4]),
                        _Box(19, 0.8, [4, 4, 7, 7]),
                    ],
                    masks=[first_mask, second_mask],
                    orig_shape=(7, 7),
                )
            ])
            embeddings = iter([
                np.array([1, 0], dtype=np.float32),
                np.array([0, 1], dtype=np.float32),
            ])
            provider._embed = lambda image: next(embeddings)
            try:
                results = provider.identify(_detection(), "source-1", multiple=True)
            finally:
                provider.close()

        self.assertIsInstance(results, list)
        self.assertEqual([result.identity_id for result in results], [
            "cow-main-0001",
            "cow-main-0002",
        ])

    def test_provider_aggregates_sampled_detections_into_one_identity_event(self):
        with tempfile.TemporaryDirectory() as directory:
            provider = WildlifeToolsIdentityProvider(
                IdentityProviderConfig(
                    id="cow-main",
                    database=Path(directory) / "identities.sqlite",
                    create_after=1,
                )
            )
            embeddings = iter([
                np.array([1, 0], dtype=np.float32),
                np.array([0.9, 0.1], dtype=np.float32),
                np.array([0.8, 0.2], dtype=np.float32),
            ])
            provider._embed = lambda image: next(embeddings)
            try:
                result = provider.identify(
                    [_detection(), _detection(), _detection()],
                    "source-1",
                )
                sample_count = provider.store.connection.execute(
                    "SELECT COUNT(*) FROM samples"
                ).fetchone()[0]
            finally:
                provider.close()

        self.assertIsInstance(result, IdentityResult)
        self.assertEqual(result.status, "created")
        self.assertEqual(sample_count, 1)

    def test_provider_uses_tracked_segment_crops_for_sampled_event(self):
        first_track_mask = np.zeros((7, 7), dtype=bool)
        first_track_mask[0:4, 0:4] = True
        second_track_mask = np.zeros((7, 7), dtype=bool)
        second_track_mask[4:7, 4:7] = True

        with tempfile.TemporaryDirectory() as directory:
            provider = WildlifeToolsIdentityProvider(
                IdentityProviderConfig(
                    id="cow-main",
                    database=Path(directory) / "identities.sqlite",
                    segment_model="seg.pt",
                    segment_confidence=0.5,
                    create_after=1,
                    crop_padding=0,
                )
            )
            provider.segmenter = _Segmenter([
                _SegmentationResult(
                    names={19: "cow"},
                    boxes=[
                        _Box(19, 0.9, [0, 0, 4, 4]),
                        _Box(19, 0.8, [4, 4, 7, 7]),
                    ],
                    masks=[first_track_mask, second_track_mask],
                    track_ids=[7, 8],
                    orig_shape=(7, 7),
                ),
                _SegmentationResult(
                    names={19: "cow"},
                    boxes=[_Box(19, 0.85, [0, 0, 4, 4])],
                    masks=[first_track_mask],
                    track_ids=[7],
                    orig_shape=(7, 7),
                ),
            ])
            embedded_shapes = []
            embeddings = iter([
                np.array([1, 0], dtype=np.float32),
                np.array([0.9, 0.1], dtype=np.float32),
            ])

            def embed(image):
                embedded_shapes.append(image.shape)
                return next(embeddings)

            provider._embed = embed
            try:
                result = provider.identify(
                    [
                        _detection(date=datetime(2026, 1, 1, 12, 0, 0)),
                        _detection(date=datetime(2026, 1, 1, 12, 0, 1)),
                    ],
                    "source-1",
                )
                sample_count = provider.store.connection.execute(
                    "SELECT COUNT(*) FROM samples"
                ).fetchone()[0]
            finally:
                provider.close()

        self.assertIsInstance(result, IdentityResult)
        self.assertEqual(result.status, "created")
        self.assertEqual(sample_count, 1)
        self.assertEqual(embedded_shapes, [(4, 4, 3), (4, 4, 3)])

    def test_provider_falls_back_to_best_frame_when_tracking_fails(self):
        mask = np.zeros((7, 7), dtype=bool)
        mask[1:5, 1:5] = True

        with tempfile.TemporaryDirectory() as directory:
            provider = WildlifeToolsIdentityProvider(
                IdentityProviderConfig(
                    id="cow-main",
                    database=Path(directory) / "identities.sqlite",
                    segment_model="seg.pt",
                    segment_confidence=0.5,
                    create_after=1,
                    crop_padding=0,
                )
            )
            provider.segmenter = _FailingTrackSegmenter([
                _SegmentationResult(
                    names={19: "cow"},
                    boxes=[_Box(19, 0.9, [1, 1, 5, 5])],
                    masks=[mask],
                    orig_shape=(7, 7),
                )
            ])
            embedded_shapes = []

            def embed(image):
                embedded_shapes.append(image.shape)
                return np.array([1, 0], dtype=np.float32)

            provider._embed = embed
            try:
                result = provider.identify(
                    [
                        _detection(
                            date=datetime(2026, 1, 1, 12, 0, 0),
                            confidence={"cow": 0.7},
                        ),
                        _detection(
                            date=datetime(2026, 1, 1, 12, 0, 1),
                            confidence={"cow": 0.9},
                        ),
                    ],
                    "source-1",
                )
            finally:
                provider.close()

        self.assertIsInstance(result, IdentityResult)
        self.assertEqual(result.status, "created")
        self.assertEqual(embedded_shapes, [(4, 4, 3)])

    def test_model_signature_includes_segment_settings(self):
        config = IdentityProviderConfig(
            id="cow-main",
            database=Path("identities.sqlite"),
            segment_model="yolo11n-seg.pt",
            segment_labels=["cow"],
        )

        self.assertIn("segment_model=yolo11n-seg.pt", _model_signature(config))
        self.assertIn("segment_labels=cow", _model_signature(config))
        self.assertIn("segment_crop=mask", _model_signature(config))
        self.assertNotIn("segment_selection", _model_signature(config))


class ExporterIdentityTests(unittest.TestCase):
    def test_telegram_caption_includes_identity(self):
        exporter = TelegramExporter(
            token="test-token",
            chat="test-chat",
            confidence=0,
            alert_every=1,
            include_video=False,
            include_image=True,
            include_plot=False,
            include_crop=False,
            video_width=None,
        )
        detection = _detection(
            identity=IdentityResult(
                provider="cow-main",
                identity_id="cow-main-0001",
                name=None,
                status="matched",
                similarity=0.82,
            )
        )

        payload = exporter.get_payload(detection, [detection], validated=True)

        self.assertIn("Identity: cow-main-0001 (82%)", payload["media"])

    def test_webhook_payload_includes_identity(self):
        exporter = WebhookExporter(
            url="https://example.com",
            token=None,
            confidence=0,
            data_type="binary",
            data_max=None,
            include_video=False,
            include_image=False,
            include_plot=False,
            include_crop=False,
            video_width=None,
        )
        detection = _detection(
            identity=IdentityResult(
                provider="cow-main",
                identity_id="cow-main-0001",
                name=None,
                status="matched",
                similarity=0.82,
            )
        )

        payload = exporter.get_payload(detection, [detection], validated=True)

        self.assertEqual(
            payload["identity"],
            {
                "provider": "cow-main",
                "identity_id": "cow-main-0001",
                "name": None,
                "status": "matched",
                "similarity": 0.82,
            },
        )


def _detection(
    label: str | None = "cow",
    confidence: dict[str, float] | None = None,
    identity: IdentityResult | None = None,
    date: datetime | None = None,
) -> Detection:
    return Detection(
        date=date or datetime(2026, 1, 1, 12, 0, 0),
        images=ImageSet(
            jpg=np.zeros((10, 10, 3), dtype=np.uint8),
            crop=Crop(1, 1, 8, 8, label=label, confidence=0.9) if label else None,
        ),
        confidence=confidence or ({label: 0.9} if label else {}),
        identity=identity,
    )


class _Validator:
    def __init__(self, validated: bool | None):
        self.validated = validated

    def validate(self, best_detection, detections):
        return self.validated


class _FailingIdentityService:
    calls = 0

    def identify(self, provider, detection, source, multiple=False):
        self.calls += 1
        raise RuntimeError("identity failed")


class _ListIdentityService:
    multiple = None

    def identify(self, provider, detection, source, multiple=False):
        self.multiple = multiple
        return [
            IdentityResult(
                provider=provider,
                identity_id="cow-main-0001",
                name=None,
                status="matched",
                similarity=0.9,
            ),
            IdentityResult(
                provider=provider,
                identity_id="cow-main-0002",
                name=None,
                status="matched",
                similarity=0.8,
            ),
        ]


class _RecordingIdentityService:
    detection = None

    def identify(self, provider, detection, source, multiple=False):
        self.detection = detection
        return IdentityResult(
            provider=provider,
            identity_id="cow-main-0001",
            name=None,
            status="matched",
            similarity=0.9,
        )


class _RecordingExporter:
    called = False

    def export(self, best_detection, detections, validated):
        self.called = True


def _detector_identity_config(**overrides):
    config = {
        "id": "cow-main",
        "database": Path("identities.sqlite"),
    }
    config.update(overrides)
    return DetectorIdentityConfig(**config)


def _detector_for_export(validated: bool | None):
    detector = Detector.__new__(Detector)
    detector.detections = defaultdict(list)
    detector.detections["source-1"] = [_detection()]
    detector.yolo_config = None
    detector.validator = _Validator(validated)
    detector.exporters = [_RecordingExporter()]
    detector.identity_service = _FailingIdentityService()
    detector.identity_config = _detector_identity_config()
    detector.export_executor = ThreadPoolExecutor(max_workers=1)
    return detector


class _Value:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class _Box:
    def __init__(self, class_id, confidence, xyxy):
        self.cls = _Value(class_id)
        self.conf = _Value(confidence)
        self.xyxy = [xyxy]


class _Boxes(list):
    id = None


class _MaskData:
    def __init__(self, mask):
        self.mask = np.array(mask)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.mask


class _Masks:
    def __init__(self, masks):
        self.data = [_MaskData(mask) for mask in masks]


class _SegmentationResult:
    def __init__(self, names, boxes, masks, orig_shape=(2, 2), track_ids=None):
        self.names = names
        self.boxes = _Boxes(boxes)
        self.boxes.id = (
            [_Value(track_id) for track_id in track_ids]
            if track_ids is not None
            else None
        )
        self.masks = _Masks(masks)
        self.orig_shape = orig_shape


class _Segmenter:
    def __init__(self, results):
        self.results = results

    def predict(self, **kwargs):
        return self.results

    def track(self, **kwargs):
        return self.results


class _FailingTrackSegmenter(_Segmenter):
    def track(self, **kwargs):
        raise ModuleNotFoundError("lap")


if __name__ == "__main__":
    unittest.main()
