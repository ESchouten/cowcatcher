from typing import Self

import requests

from aidetector.config import (
    Config,
    Detection,
    DetectorConfig,
    get_timestamped_filename,
)
from aidetector.exporters.exporter import Exporter


class TelegramExporter(Exporter):
    base_url: str
    telegram_chat_id: str

    def __init__(self, telegram_bot_token: str, telegram_chat_id: str):
        super().__init__(telegram_bot_token, telegram_chat_id)
        self.base_url = f"https://api.telegram.org/bot{telegram_bot_token}"
        self.telegram_chat_id = telegram_chat_id

    @classmethod
    def fromConfig(cls, config: Config, detector: DetectorConfig) -> Self | None:
        telegram_chat_id = detector.telegram_chat_id or config.telegram_chat_id
        if config.telegram_bot_token is None or telegram_chat_id is None:
            return None
        return cls(config.telegram_bot_token, telegram_chat_id)

    def export(self, sorted_detections: list[Detection]):
        if not sorted_detections:
            return
        try:
            self.logger.info(
                f"Sending photo to Telegram with confidence {sorted_detections[0].confidence}"
            )
            url = f"{self.base_url}/sendPhoto"
            files = {
                "photo": (
                    get_timestamped_filename(sorted_detections[0]),
                    sorted_detections[0].jpg,
                    "image/jpeg",
                )
            }
            payload = {
                "chat_id": self.telegram_chat_id,
                "caption": "Help de AI te verbeteren door goede detecties te beoordelen met een like!",
            }
            response = requests.post(url, data=payload, files=files)
            if response.status_code != 200:
                self.logger.error(f"Failed to send photo to Telegram: {response.text}")
        except Exception as e:
            self.logger.error(f"Error sending photo to Telegram: {e}")
