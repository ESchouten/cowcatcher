
import requests
import logging
from config import Config
from notifier import NotifierClass


class HomeAssistantNotifierClass(NotifierClass):
    def __init__(self,
                 config: Config = None,     # pyright: ignore[reportArgumentType]
                 bottoken: str = None,      # pyright: ignore[reportArgumentType]
                 chat_ids: list[str] = None # pyright: ignore[reportArgumentType]
                ):
        # HomeAssistant specifieke init
        pass
    
    def _test_connection(self) -> bool:
        # implementatie voor HomeAssistant
        return True
    
    def send_photo(self, image_path: str, caption: str, disable_notification: bool = False) -> bool:
        # implementatie voor HomeAssistant
        return True
    
    def send_message(self, message: str) -> bool:
        # implementatie voor HomeAssistant

        return True