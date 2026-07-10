import base64
import json
import logging
from collections.abc import Callable
from time import sleep
from typing import Any

import litellm
from aidetector.detection.models import Detection
from aidetector.media.video import generate_mp4, get_crop, get_image
from aidetector.utils.config import VLMConfig
from litellm.exceptions import ServiceUnavailableError

logger = logging.getLogger(__name__)
MAX_ATTEMPTS = 5
RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "detection_result",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "detected": {"type": "boolean"},
                "confidence": {"type": "number"},
                "reasoning": {"type": "string"},
            },
            "required": ["detected", "confidence", "reasoning"],
            "additionalProperties": False,
        },
    },
}


class Validator:
    def __init__(
        self,
        vlms: list[VLMConfig],
        *,
        completion: Callable[..., Any] = litellm.completion,
        sleeper: Callable[[float], None] = sleep,
    ):
        self.vlms = list(vlms)
        self._completion = completion
        self._sleep = sleeper

    def validate(
        self,
        detection: Detection,
        detections: list[Detection],
    ) -> bool | None:
        for config in self.vlms:
            messages = _messages(config, detection, detections)
            options = {
                key: value
                for key, value in {
                    "api_key": config.key,
                    "base_url": config.url,
                }.items()
                if value is not None
            }
            models = [config.model] if isinstance(config.model, str) else config.model
            for model in models:
                result = self._validate_model(model, messages, options)
                if result is not None:
                    return result
        return None

    def _validate_model(
        self,
        model: str,
        messages: list[dict[str, Any]],
        options: dict[str, str],
    ) -> bool | None:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                response = self._completion(
                    model=model,
                    messages=messages,
                    response_format=RESPONSE_FORMAT,
                    **options,
                )
                content = response.choices[0].message.content
                output = json.loads(content)
                if not isinstance(output.get("detected"), bool):
                    raise ValueError("VLM response has no boolean 'detected' field")
                logger.info("VLM %s returned %s", model, output)
                return output["detected"]
            except ServiceUnavailableError:
                if attempt == MAX_ATTEMPTS:
                    logger.warning("VLM %s remained unavailable", model)
                    return None
                logger.warning(
                    "VLM %s unavailable; retrying (%d/%d)",
                    model,
                    attempt,
                    MAX_ATTEMPTS,
                )
                self._sleep(attempt)
            except Exception:
                logger.exception("VLM %s validation failed", model)
                return None
        return None


def _messages(
    config: VLMConfig,
    detection: Detection,
    detections: list[Detection],
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": config.prompt}]
    video = (
        generate_mp4(
            detections,
            width=1280,
            plot=False,
            padding=config.crop_padding,
        )
        if config.strategy == "VIDEO"
        else None
    )
    if video is not None:
        encoded = base64.b64encode(video).decode("ascii")
        content.append(
            {
                "type": "file",
                "file": {"file_data": f"data:video/mp4;base64,{encoded}"},
            }
        )
    else:
        crop = get_crop(
            detection,
            padding=config.crop_padding,
        )
        image = crop if crop is not None else detection.images.jpg
        encoded = base64.b64encode(get_image(image)).decode("ascii")
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
            }
        )
    return [{"role": "user", "content": content}]
