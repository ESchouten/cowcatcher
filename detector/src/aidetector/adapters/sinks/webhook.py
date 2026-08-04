import base64
import json
import logging
from typing import Any

import requests

from aidetector.adapters.sinks.media import encode_media
from aidetector.adapters.sinks.metadata import DetectionMetadata
from aidetector.pipeline.messages import CompletedEvent
from aidetector.utils.config import WebhookConfig

DEFAULT_WEBHOOK_TIMEOUT = 30


class WebhookSink:
    def __init__(self, config: WebhookConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def send(self, message: CompletedEvent) -> None:
        event = message.event
        self.logger.info("Sending webhook with confidence %.3f", event.confidence)
        headers = dict(self.config.headers or {})
        if self.config.token is not None:
            headers["Authorization"] = self.config.token
        request: dict[str, Any] = {
            "headers": headers,
            "timeout": self.config.timeout or DEFAULT_WEBHOOK_TIMEOUT,
        }
        if self.config.body is not None:
            request["data"] = self.config.body
        elif self.config.data_type != "none":
            payload = DetectionMetadata.from_event(event, message.validated).as_dict()
            media = encode_media(message, self.config, data_max=self.config.data_max)
            if self.config.data_type == "base64":
                payload.update(
                    {
                        name: base64.b64encode(item.content).decode("ascii")
                        for name, item in media.items()
                    }
                )
                request["json"] = payload
            else:
                request["data"] = {
                    key: json.dumps(value) if isinstance(value, (dict, list)) else value
                    for key, value in payload.items()
                }
                request["files"] = {
                    name: item.request_file() for name, item in media.items()
                }

        response = requests.request(self.config.method, self.config.url, **request)
        response.raise_for_status()
