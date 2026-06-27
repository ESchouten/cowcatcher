import base64
import logging
from typing import Any

import requests
from aidetector.exporters.exporter import Exporter
from aidetector.media.video import compress_jpg, generate_mp4, get_crop, get_image, get_plot
from aidetector.utils.config import (
    Detection,
    WebhookConfig,
    get_timestamped_filename,
    max_confidence,
)


class WebhookExporter(Exporter[WebhookConfig]):
    logger = logging.getLogger(__name__)

    def __init__(self, config: WebhookConfig):
        super().__init__(config)

    def get_file(self, detection: Detection, detections: list[Detection]):
        if self.config.data_type in ("base64", "none"):
            return None
        files = {}
        if self.config.include_image:
            image = get_image(detection.images.jpg)
            if self.config.data_max is not None:
                compressed = compress_jpg(detection.images.jpg, self.config.data_max)
                if compressed is not None:
                    image = compressed
            files["image"] = (
                get_timestamped_filename(detection),
                image,
                "image/jpeg",
            )
        if self.config.include_plot:
            image = get_plot(detection)
            photo = get_image(image)
            if self.config.data_max is not None:
                compressed = compress_jpg(image, self.config.data_max)
                if compressed is not None:
                    photo = compressed
            files["photo"] = (
                get_timestamped_filename(detection),
                photo,
                "image/jpeg",
            )
        if self.config.include_crop and detection.images.crop_region:
            c = get_crop(detection)
            if c is not None:
                crop = get_image(c)
                if self.config.data_max is not None:
                    compressed = compress_jpg(c, self.config.data_max)
                    if compressed is not None:
                        crop = compressed
                files["crop"] = (
                    f"{get_timestamped_filename(detection).replace('.jpg', '_crop.jpg')}",
                    crop,
                    "image/jpeg",
                )
        if self.config.include_video:
            video = generate_mp4(
                detections,
                width=self.config.video_width,
                crf=self.config.video_crf,
                data_max=self.config.data_max,
                padding=self.config.crop_padding,
            )
            if video:
                files["video"] = (
                    f"{get_timestamped_filename(detection).replace('.jpg', '.mp4')}",
                    video,
                    "video/mp4",
                )
        return files

    def get_payload(
        self,
        best_detection: Detection,
        detections: list[Detection],
        validated: bool | None,
    ) -> dict[str, str | bytes]:
        data: dict = {
            "confidence": max_confidence(best_detection.confidence),
            "timestamp": best_detection.date.isoformat(),
            "duration": (detections[-1].date - detections[0].date).total_seconds(),
            "validated": validated,
        }
        if self.config.data_type == "base64":
            if self.config.include_image:
                jpg = get_image(best_detection.images.jpg)
                if self.config.data_max is not None:
                    compressed = compress_jpg(best_detection.images.jpg, self.config.data_max)
                    if compressed is not None:
                        jpg = compressed
                data["image"] = base64.b64encode(jpg).decode("utf-8")
            if self.config.include_plot:
                img = get_plot(best_detection)
                jpg = get_image(img)
                if self.config.data_max is not None:
                    compressed = compress_jpg(img, self.config.data_max)
                    if compressed is not None:
                        jpg = compressed
                data["photo"] = base64.b64encode(jpg).decode("utf-8")
            if self.config.include_crop and best_detection.images.crop_region:
                c = get_crop(best_detection)
                if c is not None:
                    jpg = get_image(c)
                    if self.config.data_max is not None:
                        compressed = compress_jpg(c, self.config.data_max)
                        if compressed is not None:
                            jpg = compressed
                    data["crop"] = base64.b64encode(jpg).decode("utf-8")
            if self.config.include_video:
                video = generate_mp4(
                    detections,
                    width=self.config.video_width,
                    crf=self.config.video_crf,
                    padding=self.config.crop_padding,
                )
                if video:
                    data["video"] = base64.b64encode(video).decode("utf-8")
        return data

    def get_headers(self):
        headers = dict(self.config.headers or {})
        if self.config.token is not None:
            headers["Authorization"] = self.config.token
        return headers

    def filtered_export(
        self,
        best_detection: Detection,
        detections: list[Detection],
        validated: bool | None,
    ):
        try:
            self.logger.info(
                "Sending webhook with confidence %s",
                max_confidence(best_detection.confidence),
            )
            headers = self.get_headers()

            new_detection = Detection(
                best_detection.date,
                best_detection.images,
                best_detection.confidence,
            )

            request: dict[str, Any] = {
                "headers": headers,
                "timeout": self.config.timeout,
            }
            if self.config.body is not None:
                request["data"] = self.config.body
            elif self.config.data_type == "base64":
                request["json"] = self.get_payload(new_detection, detections, validated)
            elif self.config.data_type == "binary":
                request["data"] = self.get_payload(new_detection, detections, validated)
                request["files"] = self.get_file(new_detection, detections)

            response = requests.request(self.config.method, self.config.url, **request)

            if response.status_code >= 400:
                self.logger.error(f"Failed to send webhook: {response.text}")
        except Exception as e:
            self.logger.error(f"Error sending webhook: {e}")
