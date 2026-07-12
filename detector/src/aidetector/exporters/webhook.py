import base64
from typing import Any

import requests

from aidetector.detection.events import DetectionEvent
from aidetector.exporters.exporter import Exporter
from aidetector.exporters.media import encode_media
from aidetector.utils.config import WebhookConfig


class WebhookExporter(Exporter[WebhookConfig]):
    def _export(
        self,
        event: DetectionEvent,
        validated: bool | None,
    ) -> None:
        self.logger.info(
            "Sending webhook with confidence %.3f",
            event.confidence,
        )
        headers = dict(self.config.headers or {})
        if self.config.token is not None:
            headers["Authorization"] = self.config.token
        request: dict[str, Any] = {
            "headers": headers,
            "timeout": self.config.timeout,
        }
        if self.config.body is not None:
            request["data"] = self.config.body
        elif self.config.data_type != "none":
            payload: dict[str, Any] = {
                "confidence": event.confidence,
                "timestamp": event.best.date.isoformat(),
                "duration": event.duration,
                "validated": validated,
            }
            media = encode_media(
                event,
                self.config,
                data_max=self.config.data_max,
            )
            if self.config.data_type == "base64":
                payload.update(
                    {
                        name: base64.b64encode(item.content).decode("ascii")
                        for name, item in media.items()
                    }
                )
                request["json"] = payload
            else:
                request["data"] = payload
                request["files"] = {
                    name: item.request_file() for name, item in media.items()
                }

        response = requests.request(
            self.config.method,
            self.config.url,
            **request,
        )
        response.raise_for_status()
