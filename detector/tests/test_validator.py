import json
from litellm.exceptions import ServiceUnavailableError

from aidetector.adapters.validation.vlm import MAX_ATTEMPTS, VLMValidator
from aidetector.media.artifacts import EventArtifacts
from aidetector.utils.config import VLMConfig
from tests.factories import make_event


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
    item = make_event()
    assert VLMValidator([]).validate(item, EventArtifacts(item)) is None


def test_validator_falls_back_to_next_model(monkeypatch):
    calls = []

    def completion(**kwargs):
        calls.append(kwargs)
        if kwargs["model"] == "first":
            raise ValueError("bad response")
        return response(True)

    validator = VLMValidator(
        [VLMConfig(prompt="A cow?", model=["first", "second"], strategy="IMAGE")],
        completion=completion,
    )

    item = make_event()
    assert validator.validate(item, EventArtifacts(item)) is True
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

    validator = VLMValidator(
        [VLMConfig(prompt="A cow?", model="model", strategy="IMAGE")],
        completion=completion,
        sleeper=sleeps.append,
    )

    item = make_event()
    assert validator.validate(item, EventArtifacts(item)) is False
    assert len(attempts) == MAX_ATTEMPTS
    assert sleeps == [1, 2, 3, 4]
