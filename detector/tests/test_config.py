import json
from pathlib import Path

from aidetector.utils.config import Config, confidence_matches, matching_confidences


def test_example_config_validates():
    repo_root = Path(__file__).resolve().parents[2]
    config_json = json.loads((repo_root / "example/config.json").read_text())

    config = Config(**config_json)

    assert len(config.detectors) == 1
    assert config.detectors[0].detection.source == ["sprong24.mp4"]
    assert config.detectors[0].exporters is not None


def test_confidence_helpers_support_global_and_per_class_thresholds():
    confidence = {"cow": 0.8, "horse": 0.3}

    assert confidence_matches(confidence, 0.75)
    assert not confidence_matches(confidence, 0.9)
    assert confidence_matches(confidence, {"horse": 0.2})
    assert matching_confidences(confidence, {"cow": 0.7, "horse": 0.7}) == ["cow"]
