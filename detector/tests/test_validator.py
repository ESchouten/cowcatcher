import json
from datetime import datetime

import numpy as np
from litellm.exceptions import ServiceUnavailableError

from aidetector.detection.models import Detection, ImageSet
from aidetector.detection.validator import MAX_ATTEMPTS, Validator
from aidetector.utils.config import VLMConfig


def detection() -> Detection:
    return Detection(
        datetime.now(),
        ImageSet(np.zeros((40, 60, 3), dtype=np.uint8)),
        {"cow": 0.9},
    )


def response(detected: bool):
    message = type(
        "Message",
        (),
        {
            "content": json.dumps(
                {"detected": detected, "confidence": 0.9, "reasoning": "test"}
            )
        },
    )()
    choice = type("Choice", (), {"message": message})()
    return type("Response", (), {"choices": [choice]})()


def test_validator_returns_none_without_models():
    item = detection()

    assert Validator([]).validate(item, [item]) is None


def test_validator_falls_back_to_next_model(monkeypatch):
    calls = []

    def completion(**kwargs):
        calls.append(kwargs)
        if kwargs["model"] == "first":
            raise ValueError("bad response")
        return response(True)

    item = detection()
    validator = Validator(
        [VLMConfig(prompt="A cow?", model=["first", "second"], strategy="IMAGE")],
        completion=completion,
    )

    assert validator.validate(item, [item]) is True
    assert [call["model"] for call in calls] == ["first", "second"]
    assert calls[-1]["messages"][0]["content"][1]["type"] == "image_url"


def test_validator_retries_temporary_unavailability():
    attempts = []
    sleeps = []

    def completion(**_kwargs):
        attempts.append(1)
        if len(attempts) < MAX_ATTEMPTS:
            raise ServiceUnavailableError("offline", "model", "provider")
        return response(False)

    item = detection()
    validator = Validator(
        [VLMConfig(prompt="A cow?", model="model", strategy="IMAGE")],
        completion=completion,
        sleeper=sleeps.append,
    )

    assert validator.validate(item, [item]) is False
    assert len(attempts) == MAX_ATTEMPTS
    assert sleeps == [1, 2, 3, 4]
