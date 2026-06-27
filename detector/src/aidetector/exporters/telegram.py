import json

from aidetector.exporters.webhook import WebhookExporter
from aidetector.media.video import generate_mp4
from aidetector.utils.config import (
    ChatConfig,
    Detection,
    WebhookConfig,
    max_confidence,
)


class TelegramExporter(WebhookExporter):
    telegram: ChatConfig
    alert_count: int

    def __init__(self, config: ChatConfig):
        self.telegram = config
        super().__init__(
            WebhookConfig(
                url=f"https://api.telegram.org/bot{config.token}/sendMediaGroup",
                token=config.token,
                confidence=config.confidence,
                crop_padding=config.crop_padding,
                export_rejected=config.export_rejected,
                data_type="binary",
                data_max=12_000_000,
                include_video=config.include_video,
                include_image=config.include_image,
                include_plot=config.include_plot,
                include_crop=config.include_crop,
                video_width=config.video_width,
                video_crf=config.video_crf,
            )
        )
        self.alert_count = 0

    def get_payload(
        self,
        best_detection: Detection,
        detections: list[Detection],
        validated: bool | None,
    ):
        self.alert_count += 1
        media = []
        if self.telegram.include_image:
            media.append(
                {
                    "type": "photo",
                    "media": "attach://image",
                }
            )
        if self.telegram.include_plot:
            media.append(
                {
                    "type": "photo",
                    "media": "attach://photo",
                }
            )
        if self.telegram.include_crop and best_detection.images.crop_region:
            media.append(
                {
                    "type": "photo",
                    "media": "attach://crop",
                }
            )

        if self.telegram.include_video:
            video = generate_mp4(
                detections,
                width=self.telegram.video_width,
                crf=self.telegram.video_crf,
                padding=self.telegram.crop_padding,
            )
            if video:
                media.append(
                    {
                        "type": "video",
                        "media": "attach://video",
                    }
                )

        thumbs = "\n👍 / 👎" if validated is None else ""
        identity = best_detection.identity
        identity_line = ""
        if len(best_detection.identities) > 1:
            identity_line = "\nIdentities: " + ", ".join(
                _format_identity(identity) for identity in best_detection.identities
            )
        elif identity is not None:
            similarity = (
                f" ({int(identity.similarity * 100)}%)"
                if identity.similarity is not None
                else ""
            )
            if identity.status == "created" and identity.identity_id is not None:
                identity_line = f"\nNew identity: {identity.identity_id}{similarity}"
            elif identity.status == "matched" and identity.identity_id is not None:
                identity_name = identity.name or identity.identity_id
                identity_line = f"\nIdentity: {identity_name}{similarity}"
            else:
                identity_line = "\nIdentity: unknown"
        media[0]["caption"] = (
            f"{int(max_confidence(best_detection.confidence) * 100)}%{' ✅' if validated else ' ❌' if validated is False else ''}\n{round((detections[-1].date - detections[0].date).total_seconds())} second(s){identity_line}{thumbs}"
        )

        return {
            "chat_id": self.telegram.chat,
            "disable_notification": self.alert_count % self.telegram.alert_every != 0,
            "media": json.dumps(media),
        }


def _format_identity(identity) -> str:
    identity_id = identity.name or identity.identity_id
    if identity_id is None:
        return "unknown"
    if identity.similarity is None:
        return identity_id
    return f"{identity_id} ({int(identity.similarity * 100)}%)"
