"""
Module beschrijving: wat doet deze module?
"""
import requests
import logging
from config import Config
from notifier import NotifierClass



class TelegramNotifierClass(NotifierClass):
    """
    Deze module verstuur berichten naar telegram

    Attributes:
        attribuut1 (type): Beschrijving van attribuut1
        attribuut2 (type): Beschrijving van attribuut2
    """
    def __init__(self,
                 config: Config = None,     # pyright: ignore[reportArgumentType]
                 bottoken: str = None,      # pyright: ignore[reportArgumentType]
                 chat_ids: list[str] = None # pyright: ignore[reportArgumentType]
                ):
        """
        Initialiseer een nieuwe instance van TelegramNotifierClass.

        Args:
            param1 (type): Beschrijving van parameter1
            param2 (type, optional): Beschrijving van parameter2. Defaults to None.
        """
        # Initialize the logger
        self.logger = logging.getLogger(__name__)
    
        # Creat an instance of Config class to read parameters
        if config is None:
            config = Config()

        notifier_config = config.get_notifier_config("telegram")

        self.timeout = notifier_config["timeout"]
        self.disable_notification = notifier_config["disable_notification"]

        # If bottoken is passed as parameter this overrides the config ones
        if bottoken is None:
            self.bottoken = notifier_config["api_key"]
        else:
            self.bottoken = bottoken
        # If chat_ids is passed as parameter this overrides the config ones
        if chat_ids is None:
            self.chat_ids = notifier_config["chat_id"]
        else:
            self.chat_ids = chat_ids


        self._base_url=f"https://api.telegram.org/bot{self.bottoken}"
        # self._private_attribuut = None  # Private attribuut (conventie)






    def _test_connection(self)->bool:
        try:
            url = f"{self._base_url}/getMe"
            response = requests.get(url)
            if response.status_code == 200:
                self.logger.info("Telegram connection successfully tested.")
                return True
            else:
                self.logger.info(f"ERROR testing Telegram connection: {response.text}")
                return False
        except Exception as e:
            self.logger.info(f"ERROR testing Telegram connection: {str(e)}")
            return False

    def send_photo(self,image_path: str, caption: str, disable_notification: bool=False)-> bool:
        """Sends a photo with caption to Telegram."""
        
        try:
            url = f"{self._base_url}/sendPhoto"
            with open(image_path, 'rb') as photo:
                files = {'photo': photo}
                for chat_id in self.chat_ids:
                    data = {
                        'chat_id': chat_id,
                        'caption': caption,
                        'disable_notification': disable_notification
                    }
                    response = requests.post(url, files=files, data=data)

                if response.status_code != 200: # pyright: ignore[reportPossiblyUnboundVariable]
                    self.logger.info(f"ERROR sending Telegram photo: {response.text}") # pyright: ignore[reportPossiblyUnboundVariable]
                    return False
            return response.json() # pyright: ignore[reportPossiblyUnboundVariable]
        except Exception as e:
            self.logger.info(f"ERROR sending Telegram photo: {str(e)}")
            return False


    def send_message(self,message: str) -> bool:
        """Sends a text message to Telegram."""

        try:
            url = f"{self._base_url}/sendMessage"
            for chat_id in self.chat_ids:
                data = { 'chat_id': chat_id
                       , 'text': message
                        }
                response = requests.post(url, data=data)

                if response.status_code != 200:
                    self.logger.info(f"ERROR sending Telegram message: {response.text}")
                    return False

            return True

        except Exception as e:
            self.logger.info(f"ERROR sending Telegram message: {str(e)}")
            return False




    def publieke_methode(self, arg):
        """
        Beschrijving van wat deze methode doet.

        Args:
            arg (type): Beschrijving van argument

        Returns:
            type: Beschrijving van return waarde

        Raises:
            ValueError: Wanneer arg invalide is
        """
        if not arg:
            raise ValueError("Argument mag niet leeg zijn")

        return f"Resultaat: {arg}"

    def _private_methode(self):
        """
        Private methode (conventie met underscore).
        Alleen bedoeld voor intern gebruik.
        """
        pass

# Testing the classV:
if __name__ == "__main__":
    logger = logging.getLogger("notifier.py - main")
    logger.info("\n=== Config Class ===")
    # Maak instance
    obj = TelegramNotifierClass("8276937928:AAFW9QByaDmOkXRdl2XvcOr5BGfB34oFfGM", ["8185578244", "8484039219"]) # type: ignore
    

    logger.info (f'result of testing method:test_connection => {obj._test_connection()}')
    logger.info (f'result of testing method:send_message => {obj.send_message("Test message")}')


