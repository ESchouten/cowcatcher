import requests
from ultralytics.engine.results import Results

from cowcatcher.config import Config, DetectorConfig
from cowcatcher.exporters.exporter import Exporter


class TelegramExporter(Exporter):
    _chat_id: str

    def __init__(self, config: Config, detector: DetectorConfig):
        super().__init__(config, detector)
        if config.telegram_bot_token is None:
            raise ValueError("Telegram bot token is not set.")
        self._base_url = f"https://api.telegram.org/bot{config.telegram_bot_token}"

        chat_id = detector.telegram_chat_id or config.telegram_chat_id
        if chat_id is None:
            raise ValueError("Telegram chat ID is not set.")
        self._chat_id = chat_id

    def export(self, data: Results):
        try:
            url = f"{self._base_url}/sendPhoto"
            files = {"photo": ("detection.jpg", data.orig_img.tobytes(), "image/jpeg")}  # pyright: ignore[reportUnknownMemberType]
            payload = {
                "chat_id": self._chat_id,
                "caption": "Help CowCatcher verbeteren door goede detecties te beoordelen met een like!",
            }
            response = requests.post(url, data=payload, files=files)
            if response.status_code != 200:
                print(f"Failed to send photo to Telegram: {response.text}")
        except Exception as e:
            print(f"Error sending photo to Telegram: {e}")
