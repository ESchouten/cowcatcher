import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from aidetector.domain.detections import confidence_matches, matching_confidences
from aidetector.utils.config import Config, IdentityConfig, YoloConfig, load_config


def identity_fragment() -> dict:
    return {
        "target_label": "cow",
        "display": {
            "singular": "cow",
            "plural": "cows",
            "official_id_label": "Cow ID",
        },
        "database": "identities/cows.sqlite",
        "candidate_filter": {
            "min_area_ratio": 0.005,
            "max_area_ratio": 0.3,
            "frame_edge_margin": 0.2,
        },
        "controlled_zone": {
            "zone_id": "identity_observation",
            "x1": 0.2,
            "y1": 0.2,
            "x2": 0.8,
            "y2": 0.8,
            "minimum_box_inside_ratio": 0.9,
            "minimum_stable_frames": 2,
            "clear_frames": 2,
        },
        "encoder": "miewid-dual-crop-v1",
        "similarity_threshold": 0.75,
        "similarity_margin": 0.05,
        "query_frames": 2,
        "gallery_frames": 4,
        "track_max_age": 10,
    }


def test_example_config_validates():
    repo_root = Path(__file__).resolve().parents[2]
    config = load_config(repo_root / "example/config.json")

    assert len(config.detectors) == 1
    assert config.detectors[0].detection.source == ["sprong24.mp4"]
    assert config.detectors[0].exporters is not None


def test_yolo_defaults_are_deterministic():
    config = YoloConfig(model="model.pt")

    assert config.frames_min == 3
    assert config.tracker == "bytetrack.yaml"


@pytest.mark.parametrize("field", list(identity_fragment()))
def test_identity_policy_fields_are_required(field):
    fragment = identity_fragment()
    del fragment[field]

    with pytest.raises(ValidationError):
        IdentityConfig(**fragment)


def test_cow_identity_preset_contains_domain_tuning():
    repo_root = Path(__file__).resolve().parents[2]
    preset = json.loads((repo_root / "config/identity/cow.json").read_text())

    config = IdentityConfig(**preset)

    assert preset == identity_fragment()
    assert config.target_label == "cow"
    assert config.database == Path("identities/cows.sqlite")
    assert config.encoder == "miewid-dual-crop-v1"
    assert config.candidate_filter.min_area_ratio == 0.005
    assert config.candidate_filter.max_area_ratio == 0.3
    assert config.candidate_filter.frame_edge_margin == 0.2
    assert config.controlled_zone.zone_id == "identity_observation"
    assert config.controlled_zone.minimum_box_inside_ratio == 0.9
    assert config.controlled_zone.minimum_stable_frames == 2
    assert config.controlled_zone.clear_frames == 2


def test_cow_detector_preset_only_contains_detector_defaults():
    repo_root = Path(__file__).resolve().parents[2]
    preset = json.loads((repo_root / "config/detector/cow.json").read_text())

    assert preset == {
        "detection": {"interval": 1},
        "yolo": {
            "model": "yolo26m-seg.pt",
            "task": "segment",
            "tracking": True,
            "confidence": {"cow": 0.1},
            "imgsz": 640,
            "iou": 0.5,
        },
    }
    assert "identity" not in preset
    assert "exporters" not in preset
    assert "source" not in preset["detection"]


def test_removed_identity_detector_presets_are_absent():
    repo_root = Path(__file__).resolve().parents[2]

    assert not (repo_root / "config/detector/cow-identity.json").exists()
    assert not (repo_root / "config/detector/cow-identity-enrollment.json").exists()


def test_identity_database_is_resolved_from_config_directory(tmp_path):
    path = tmp_path / "farm" / "config.json"
    path.parent.mkdir()
    path.write_text(
        json.dumps(
            {
                "detectors": [
                    {
                        "detection": {"source": "camera"},
                        "yolo": {
                            "model": "model.pt",
                            "tracking": True,
                            "confidence": {"cow": 0.1},
                        },
                        "identity": identity_fragment(),
                    }
                ]
            }
        )
    )

    config = load_config(path)

    assert config.detectors[0].identity is not None
    assert (
        config.detectors[0].identity.database
        == (path.parent / "identities/cows.sqlite").resolve()
    )
    assert config.detectors[0].identity.data_directory == path.parent.resolve()


def test_identity_requires_tracking_and_enabled_target_label():
    fragment = identity_fragment()

    with pytest.raises(ValidationError, match="requires YOLO tracking"):
        Config(
            detectors=[
                {
                    "detection": {"source": "camera"},
                    "yolo": {"model": "model.pt", "tracking": False},
                    "identity": fragment,
                }
            ]
        )

    with pytest.raises(ValidationError, match="target label"):
        Config(
            detectors=[
                {
                    "detection": {"source": "camera"},
                    "yolo": {
                        "model": "model.pt",
                        "tracking": True,
                        "confidence": {"horse": 0.1},
                    },
                    "identity": fragment,
                }
            ]
        )


def test_miewid_frame_counts_are_frozen():
    fragment = identity_fragment()
    fragment["query_frames"] = 3

    with pytest.raises(ValidationError, match="exactly two query frames"):
        IdentityConfig(**fragment)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("x2", 0.2, "positive extent"),
        ("minimum_box_inside_ratio", 0, "greater than 0"),
        ("minimum_stable_frames", 0, "greater than 0"),
        ("clear_frames", 0, "greater than 0"),
    ],
)
def test_controlled_identity_zone_fails_closed(
    field: str,
    value: float | int,
    message: str,
) -> None:
    fragment = identity_fragment()
    fragment["controlled_zone"][field] = value

    with pytest.raises(ValidationError, match=message):
        IdentityConfig(**fragment)


@pytest.mark.parametrize("field", list(identity_fragment()["controlled_zone"]))
def test_controlled_identity_zone_fields_are_required(field: str) -> None:
    fragment = identity_fragment()
    del fragment["controlled_zone"][field]

    with pytest.raises(ValidationError):
        IdentityConfig(**fragment)


def test_confidence_helpers_support_global_and_per_class_thresholds():
    confidence = {"cow": 0.8, "horse": 0.3}

    assert confidence_matches(confidence, 0.75)
    assert not confidence_matches(confidence, 0.9)
    assert confidence_matches(confidence, {"horse": 0.2})
    assert matching_confidences(confidence, {"cow": 0.7, "horse": 0.7}) == ["cow"]


def test_load_config_does_not_rewrite_user_file(tmp_path):
    path = tmp_path / "config.json"
    content = '{"detectors":[{"detection":{"source":"video.mp4"}}]}\n'
    path.write_text(content)

    config = load_config(path)

    assert config.detectors[0].detection.source == "video.mp4"
    assert path.read_text() == content


@pytest.mark.parametrize(
    "fragment",
    [
        {"detection": {"source": [], "interval": 0}},
        {"detection": {"source": ["camera"], "interval": -1}},
        {
            "detection": {"source": ["camera"]},
            "exporters": {"sse": {"port": 70000}},
        },
        {
            "detection": {"source": ["camera"]},
            "exporters": {
                "telegram": {"token": "token", "chat": "chat", "alert_every": 0}
            },
        },
        {
            "detection": {"source": "camera"},
            "identity": identity_fragment(),
        },
        {
            "detection": {"source": "camera"},
            "yolo": {
                "model": "model.pt",
                "tracking": True,
                "confidence": {"cow": 0.1},
            },
            "identity": {
                **identity_fragment(),
                "candidate_filter": {
                    "min_area_ratio": 0.5,
                    "max_area_ratio": 0.1,
                    "frame_edge_margin": 0.2,
                },
            },
        },
    ],
)
def test_config_rejects_invalid_runtime_values(fragment):
    with pytest.raises(ValidationError):
        Config(detectors=[fragment])


def test_config_rejects_unknown_fields():
    with pytest.raises(ValidationError, match="Unexpected keyword argument"):
        Config(detectors=[{"detection": {"source": "camera", "intervall": 1}}])
