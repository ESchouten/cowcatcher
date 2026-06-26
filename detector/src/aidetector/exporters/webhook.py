import base64
import logging
from typing import Any, Literal

import requests
from aidetector.exporters.exporter import Exporter
from aidetector.media.video import compress_jpg, generate_mp4, get_crop, get_image, get_plot
from aidetector.utils.config import (
    Confidence,
    Config,
    Detection,
    DetectorConfig,
    HttpMethod,
    WebhookConfig,
    get_timestamped_filename,
    max_confidence,
)
from typing_extensions import Self


class WebhookExporter(Exporter[WebhookConfig]):
    url: str
    method: HttpMethod
    token: str | None
    headers: dict[str, str] | None
    body: str | None
    timeout: int | None
    data_type: Literal["binary", "base64", "none"]
    include_video: bool
    include_image: bool
    include_plot: bool
    include_crop: bool
    video_width: int | None
    video_crf: int
    logger = logging.getLogger(__name__)

    def __init__(
        self,
        url: str,
        method: HttpMethod,
        token: str | None,
        headers: dict[str, str] | None,
        body: str | None,
        timeout: int | None,
        confidence: float | Confidence,
        data_type: Literal["binary", "base64", "none"],
        data_max: int | None,
        include_video: bool,
        include_image: bool,
        include_plot: bool,
        include_crop: bool,
        video_width: int | None,
        video_crf: int = 28,
        export_rejected: bool = False,
        crop_padding: float = 0.1,
    ):
        super().__init__(
            confidence,
            export_rejected,
            url,
            method,
            token,
            headers,
            body,
            timeout,
            data_type,
            data_max,
            include_video,
            include_image,
            include_plot,
            include_crop,
            video_width,
            video_crf,
            export_rejected,
            crop_padding,
        )
        self.confidence = confidence
        self.url = url
        self.method = method
        self.token = token
        self.headers = headers
        self.body = body
        self.timeout = timeout
        self.data_type = data_type
        self.data_max = data_max
        self.include_video = include_video
        self.include_image = include_image
        self.include_plot = include_plot
        self.include_crop = include_crop
        self.video_width = video_width
        self.video_crf = video_crf
        self.crop_padding = crop_padding
        self.logger = logging.getLogger(self.__class__.__name__)

    @classmethod
    def from_config(
        cls, config: Config, detector: DetectorConfig, exporter: WebhookConfig
    ) -> Self:
        return cls(
            exporter.url,
            exporter.method,
            exporter.token,
            exporter.headers,
            exporter.body,
            exporter.timeout,
            confidence=exporter.confidence or 0,
            data_type=exporter.data_type,
            data_max=exporter.data_max,
            include_video=exporter.include_video,
            include_image=exporter.include_image,
            include_plot=exporter.include_plot,
            include_crop=exporter.include_crop,
            video_width=exporter.video_width,
            video_crf=exporter.video_crf,
            export_rejected=exporter.export_rejected,
            crop_padding=exporter.crop_padding,
        )

    def get_file(self, detection: Detection, detections: list[Detection]):
        if self.data_type in ("base64", "none"):
            return None
        files = {}
        if self.include_image:
            image = get_image(detection.images.jpg)
            if self.data_max is not None:
                compressed = compress_jpg(detection.images.jpg, self.data_max)
                if compressed is not None:
                    image = compressed
            files["image"] = (
                get_timestamped_filename(detection),
                image,
                "image/jpeg",
            )
        if self.include_plot:
            image = get_plot(detection)
            photo = get_image(image)
            if self.data_max is not None:
                compressed = compress_jpg(image, self.data_max)
                if compressed is not None:
                    photo = compressed
            files["photo"] = (
                get_timestamped_filename(detection),
                photo,
                "image/jpeg",
            )
        if self.include_crop and detection.images.crop_region:
            c = get_crop(detection)
            if c is not None:
                crop = get_image(c)
                if self.data_max is not None:
                    compressed = compress_jpg(c, self.data_max)
                    if compressed is not None:
                        crop = compressed
                files["crop"] = (
                    f"{get_timestamped_filename(detection).replace('.jpg', '_crop.jpg')}",
                    crop,
                    "image/jpeg",
                )
        if self.include_video:
            video = generate_mp4(
                detections,
                width=self.video_width,
                crf=self.video_crf,
                data_max=self.data_max,
                padding=self.crop_padding,
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
        if self.data_type == "base64":
            if self.include_image:
                jpg = get_image(best_detection.images.jpg)
                if self.data_max is not None:
                    compressed = compress_jpg(best_detection.images.jpg, self.data_max)
                    if compressed is not None:
                        jpg = compressed
                data["image"] = base64.b64encode(jpg).decode("utf-8")
            if self.include_plot:
                img = get_plot(best_detection)
                jpg = get_image(img)
                if self.data_max is not None:
                    compressed = compress_jpg(img, self.data_max)
                    if compressed is not None:
                        jpg = compressed
                data["photo"] = base64.b64encode(jpg).decode("utf-8")
            if self.include_crop and best_detection.images.crop_region:
                c = get_crop(best_detection)
                if c is not None:
                    jpg = get_image(c)
                    if self.data_max is not None:
                        compressed = compress_jpg(c, self.data_max)
                        if compressed is not None:
                            jpg = compressed
                    data["crop"] = base64.b64encode(jpg).decode("utf-8")
            if self.include_video:
                video = generate_mp4(
                    detections,
                    width=self.video_width,
                    crf=self.video_crf,
                    padding=self.crop_padding,
                )
                if video:
                    data["video"] = base64.b64encode(video).decode("utf-8")
        return data

    def get_headers(self):
        headers = dict(self.headers or {})
        if self.token is not None:
            headers["Authorization"] = self.token
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

            request: dict[str, Any] = {"headers": headers, "timeout": self.timeout}
            if self.body is not None:
                request["data"] = self.body
            elif self.data_type == "base64":
                request["json"] = self.get_payload(new_detection, detections, validated)
            elif self.data_type == "binary":
                request["data"] = self.get_payload(new_detection, detections, validated)
                request["files"] = self.get_file(new_detection, detections)

            response = requests.request(self.method, self.url, **request)

            if response.status_code >= 400:
                self.logger.error(f"Failed to send webhook: {response.text}")
        except Exception as e:
            self.logger.error(f"Error sending webhook: {e}")
