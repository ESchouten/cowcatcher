import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from aidetector.domain.detections import confidence_matches, matching_confidences
from aidetector.utils.config import Config, IdentityConfig, YoloConfig, load_config


def identity_fragment() -> dict:
    return {
        "target_label": "cow",
        "database": "identities/cows.sqlite",
        "margin": 0.2,
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


def test_cow_identity_preset_is_minimal():
    repo_root = Path(__file__).resolve().parents[2]
    preset = json.loads((repo_root / "config/identity/cow.json").read_text())

    config = IdentityConfig(**preset)

    assert preset == identity_fragment()
    assert config.target_label == "cow"
    assert config.database == Path("identities/cows.sqlite")
    assert config.margin == 0.2


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


def test_identity_margin_must_leave_a_zone() -> None:
    fragment = identity_fragment()
    fragment["margin"] = 0.5

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
    ],
)
def test_config_rejects_invalid_runtime_values(fragment):
    with pytest.raises(ValidationError):
        Config(detectors=[fragment])


def test_config_rejects_unknown_fields():
    with pytest.raises(ValidationError, match="Unexpected keyword argument"):
        Config(detectors=[{"detection": {"source": "camera", "intervall": 1}}])
