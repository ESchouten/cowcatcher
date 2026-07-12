from pathlib import Path

import pytest
from pydantic import ValidationError

from aidetector.domain.detections import confidence_matches, matching_confidences
from aidetector.utils.config import Config, YoloConfig, load_config


def test_example_config_validates():
    repo_root = Path(__file__).resolve().parents[2]
    config = load_config(repo_root / "example/config.json")

    assert len(config.detectors) == 1
    assert config.detectors[0].detection.source == ["sprong24.mp4"]
    assert config.detectors[0].exporters is not None


def test_yolo_defaults_are_deterministic():
    assert YoloConfig(model="model.pt").frames_min == 3


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
            "identity": {"database": "identities/cows.sqlite"},
        },
        {
            "detection": {"source": "camera"},
            "yolo": {"model": "model.pt"},
            "identity": {
                "database": "identities/cows.sqlite",
                "min_area_ratio": 0.5,
                "max_area_ratio": 0.1,
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
