import json

from aidetector.exporters.exporter import Exporter
from aidetector.exporters.webhook import WebhookExporter
from aidetector.media.video import generate_mp4
from aidetector.utils.config import (
    ChatConfig,
    Confidence,
    Config,
    Detection,
    DetectorConfig,
    max_confidence,
)
from typing_extensions import Self


class TelegramExporter(WebhookExporter, Exporter[ChatConfig]):
    chat: str
    alert_every: int
    alert_count: int
    include_video: bool
    include_image: bool
    include_plot: bool
    include_crop: bool
    video_width: int | None
    video_crf: int

    def __init__(
        self,
        token: str,
        chat: str,
        confidence: float | Confidence,
        alert_every: int,
        include_video: bool,
        include_image: bool,
        include_plot: bool,
        include_crop: bool,
        video_width: int | None,
        video_crf: int = 28,
        export_rejected: bool = False,
        crop_padding: float = 0.1,
    ):
        url = f"https://api.telegram.org/bot{token}/sendMediaGroup"
        super().__init__(
            url,
            token,
            confidence,
            "binary",
            12_000_000,
            include_video,
            include_image,
            include_plot,
            include_crop,
            video_width,
            video_crf,
            export_rejected,
            crop_padding,
        )
        self.chat = chat
        self.alert_every = alert_every
        self.include_video = include_video
        self.include_image = include_image
        self.include_plot = include_plot
        self.include_crop = include_crop
        self.video_width = video_width
        self.video_crf = video_crf
        self.alert_count = 0
        self.crop_padding = crop_padding

    @classmethod
    def from_config(
        cls, config: Config, detector: DetectorConfig, exporter: ChatConfig
    ) -> Self:  # ty:ignore[invalid-method-override]
        return cls(
            exporter.token,
            exporter.chat,
            confidence=exporter.confidence or 0,
            alert_every=exporter.alert_every,
            include_video=exporter.include_video,
            include_image=exporter.include_image,
            include_plot=exporter.include_plot,
            include_crop=exporter.include_crop,
            video_width=exporter.video_width,
            video_crf=exporter.video_crf,
            export_rejected=exporter.export_rejected,
            crop_padding=exporter.crop_padding,
        )

    def get_payload(
        self,
        best_detection: Detection,
        detections: list[Detection],
        validated: bool | None,
    ):
        self.alert_count += 1
        media = []
        if self.include_image:
            media.append(
                {
                    "type": "photo",
                    "media": "attach://image",
                }
            )
        if self.include_plot:
            media.append(
                {
                    "type": "photo",
                    "media": "attach://photo",
                }
            )
        if self.include_crop and best_detection.images.crop:
            media.append(
                {
                    "type": "photo",
                    "media": "attach://crop",
                }
            )

        if self.include_video:
            video = generate_mp4(
                detections,
                width=self.video_width,
                crf=self.video_crf,
                padding=self.crop_padding,
            )
            if video:
                media.append(
                    {
                        "type": "video",
                        "media": "attach://video",
                    }
                )

        thumbs = "\n👍 / 👎" if validated is None else ""
        media[0]["caption"] = (
            f"{int(max_confidence(best_detection.confidence) * 100)}%{' ✅' if validated else ' ❌' if validated is False else ''}\n{round((detections[-1].date - detections[0].date).total_seconds())} second(s){thumbs}"
        )

        return {
            "chat_id": self.chat,
            "disable_notification": self.alert_count % self.alert_every != 0,
            "media": json.dumps(media),
        }
