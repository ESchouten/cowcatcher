import base64
import json
import logging
import time

import litellm
from aidetector.media.video import generate_mp4, get_crop, get_image
from aidetector.utils.config import Detection, VLMConfig
from litellm.exceptions import ServiceUnavailableError
from typing_extensions import Self


class Validator:
    logger = logging.getLogger(__name__)
    vlms: list[VLMConfig]

    def __init__(self, vlms: VLMConfig | list[VLMConfig]):
        self.vlms = [vlms] if isinstance(vlms, VLMConfig) else vlms

    @classmethod
    def from_config(cls, vlm_config: VLMConfig | list[VLMConfig]) -> Self:
        return cls(vlm_config)

    def validate(
        self, detection: Detection, detections: list[Detection]
    ) -> bool | None:
        for vlm_config in self.vlms:
            crop = get_crop(detection)
            image_url = f"data:image/jpeg;base64,{base64.b64encode(get_image(crop if crop is not None else detection.images.jpg)).decode('utf-8')}"
            video_url = None
            if vlm_config.strategy == "VIDEO":
                video = generate_mp4(
                    detections, width=1280, plot=False, padding=vlm_config.crop_padding
                )
                video_url = (
                    f"data:video/mp4;base64,{base64.b64encode(video).decode('utf-8')}"
                    if video is not None
                    else None
                )
            prompt = vlm_config.prompt

            # Build content list, excluding None values
            content = [{"type": "text", "text": prompt}]
            if video_url:
                # litellm uses 'file' type with 'file_data' for inline video/audio data
                content.append({"type": "file", "file": {"file_data": video_url}})
            else:
                content.append({"type": "image_url", "image_url": {"url": image_url}})

            messages = [{"role": "user", "content": content}]

            response_format = {
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

            kwargs = {}
            if vlm_config.key:
                kwargs["api_key"] = vlm_config.key
            if vlm_config.url:
                kwargs["base_url"] = vlm_config.url

            models = (
                [vlm_config.model]
                if isinstance(vlm_config.model, str)
                else vlm_config.model
            )

            for model in models:
                for attempt in range(5):
                    try:
                        response = litellm.completion(
                            model=model,
                            messages=messages,
                            response_format=response_format,
                            # max_tokens=1000,
                            **kwargs,
                        )
                        output = json.loads(response.choices[0].message.content)
                        self.logger.info(f"VLM detected {output}")
                        return output["detected"]
                    except ServiceUnavailableError:
                        self.logger.warning(
                            f"Model {model} unavailable, retrying ({attempt + 1}/5)..."
                        )
                        time.sleep(attempt)
                    except Exception as e:
                        self.logger.error(f"Failed to validate with model {model}: {e}")
                        break
