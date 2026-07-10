import base64
import logging
from typing import Any

import requests

from aidetector.detection.models import Detection, max_confidence
from aidetector.exporters.exporter import Exporter
from aidetector.exporters.media import MediaOptions, encode_media
from aidetector.utils.config import WebhookConfig


class WebhookExporter(Exporter[WebhookConfig]):
    logger = logging.getLogger(__name__)

    def _media_options(self) -> MediaOptions:
        return MediaOptions(
            self.config.include_image,
            self.config.include_plot,
            self.config.include_crop,
            self.config.include_video,
            self.config.video_width,
            self.config.video_crf,
            self.config.crop_padding,
            self.config.data_max,
        )

    def get_files(
        self,
        detection: Detection,
        detections: list[Detection],
    ) -> dict[str, tuple[str, bytes, str]] | None:
        if self.config.data_type != "binary":
            return None
        return {
            name: item.request_file()
            for name, item in encode_media(
                detection,
                detections,
                self._media_options(),
            ).items()
        }

    def get_payload(
        self,
        best_detection: Detection,
        detections: list[Detection],
        validated: bool | None,
    ) -> dict[str, Any]:
        data: dict[str, Any] = {
            "confidence": max_confidence(best_detection.confidence),
            "timestamp": best_detection.date.isoformat(),
            "duration": (detections[-1].date - detections[0].date).total_seconds(),
            "validated": validated,
        }
        if self.config.data_type == "base64":
            data.update(
                {
                    name: base64.b64encode(item.content).decode("ascii")
                    for name, item in encode_media(
                        best_detection,
                        detections,
                        self._media_options(),
                    ).items()
                }
            )
        return data

    def get_headers(self) -> dict[str, str]:
        headers = dict(self.config.headers or {})
        if self.config.token is not None:
            headers["Authorization"] = self.config.token
        return headers

    def _export(
        self,
        best_detection: Detection,
        detections: list[Detection],
        validated: bool | None,
    ) -> None:
        self.logger.info(
            "Sending webhook with confidence %.3f",
            max_confidence(best_detection.confidence),
        )
        request: dict[str, Any] = {
            "headers": self.get_headers(),
            "timeout": self.config.timeout,
        }
        if self.config.body is not None:
            request["data"] = self.config.body
        elif self.config.data_type == "base64":
            request["json"] = self.get_payload(
                best_detection,
                detections,
                validated,
            )
        elif self.config.data_type == "binary":
            request["data"] = self.get_payload(
                best_detection,
                detections,
                validated,
            )
            request["files"] = self.get_files(best_detection, detections)

        response = requests.request(
            self.config.method,
            self.config.url,
            **request,
        )
        response.raise_for_status()
