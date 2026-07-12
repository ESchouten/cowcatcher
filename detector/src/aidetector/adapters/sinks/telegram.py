import json
from typing import Any

import requests

from aidetector.adapters.sinks.media import EncodedFile, encode_media
from aidetector.pipeline.messages import CompletedEvent
from aidetector.utils.config import ChatConfig

TELEGRAM_API = "https://api.telegram.org"


class TelegramSink:
    def __init__(self, config: ChatConfig):
        self.config = config
        self._alert_count = 0

    def send(self, message: CompletedEvent) -> None:
        files = encode_media(
            message,
            self.config,
            data_max=12_000_000,
        )
        caption = self._caption(message)
        silent = self._next_alert_is_silent()
        if not files:
            self._request(
                "sendMessage",
                data={
                    "chat_id": self.config.chat,
                    "text": caption,
                    "disable_notification": silent,
                },
            )
        elif len(files) == 1:
            name, item = next(iter(files.items()))
            self._send_single(name, item, caption, silent)
        else:
            self._send_group(files, caption, silent)

    def _send_single(
        self,
        name: str,
        item: EncodedFile,
        caption: str,
        silent: bool,
    ) -> None:
        is_video = item.content_type == "video/mp4"
        field = "video" if is_video else "photo"
        method = "sendVideo" if is_video else "sendPhoto"
        self._request(
            method,
            data={
                "chat_id": self.config.chat,
                "caption": caption,
                "disable_notification": silent,
            },
            files={field: item.request_file()},
        )

    def _send_group(
        self,
        files: dict[str, EncodedFile],
        caption: str,
        silent: bool,
    ) -> None:
        attachments = list(files.items())[:10]
        media = [
            {
                "type": "video" if item.content_type == "video/mp4" else "photo",
                "media": f"attach://{name}",
                **({"caption": caption} if index == 0 else {}),
            }
            for index, (name, item) in enumerate(attachments)
        ]
        self._request(
            "sendMediaGroup",
            data={
                "chat_id": self.config.chat,
                "disable_notification": silent,
                "media": json.dumps(media),
            },
            files={name: item.request_file() for name, item in attachments},
        )

    def _request(
        self,
        method: str,
        *,
        data: dict[str, Any],
        files: dict[str, tuple[str, bytes, str]] | None = None,
    ) -> None:
        response = requests.post(
            f"{TELEGRAM_API}/bot{self.config.token}/{method}",
            data=data,
            files=files,
            timeout=self.config.timeout,
        )
        response.raise_for_status()

    def _next_alert_is_silent(self) -> bool:
        self._alert_count += 1
        return self._alert_count % self.config.alert_every != 0

    @staticmethod
    def _caption(message: CompletedEvent) -> str:
        event = message.event
        status = "" if message.status == "unvalidated" else f" {message.status}"
        feedback = (
            "\nApprove or reject this detection."
            if message.status == "unvalidated"
            else ""
        )
        return (
            f"{event.confidence:.0%}{status}\n"
            f"{round(event.duration)} second(s){feedback}"
        )
