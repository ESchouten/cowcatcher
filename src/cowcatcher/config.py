 # ============================================================================
 # Config class (aanbevolen voor grotere projecten)
 # ============================================================================
import json
import logging
import configparser
from typing import Dict, Any
from pathlib import Path

class Config:
    """
    Config class die verschillende bronnen kan inlezen.
    """

    def __init__(self, config_path: str = "config.json"):
        """
        Initialiseer Config en laad parameters.

        Args:
            config_path: Pad naar config bestand
        """
        # Initialize the logger
        self.logger = logging.getLogger(__name__)

        self.config_path = config_path
        self._config = self._laad_config()

    def _laad_config(self) -> Dict[str, Any]:
        """Laad config op basis van bestandstype."""
        path = Path(self.config_path)

        if path.suffix == '.json':
            # return self.laad_config_json(self.config_path)
            return self.laad_config_json()
        elif path.suffix == '.ini':
            config = self.laad_config_ini()
            # Converteer naar dict
            return {section: dict(config[section]) for section in config.sections()}
        else:
            raise ValueError(f"Onondersteund config formaat: {path.suffix}")

    def laad_config_json(self) -> Dict[str, Any]:
        """
        Laad configuratie uit een JSON bestand.

        Args:
            config_path: Pad naar het JSON config bestand

        Returns:
            Dictionary met configuratie parameters
        """
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Config bestand niet gevonden: {self.config_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Ongeldige JSON in config bestand: {e}")

    def laad_config_ini(self) -> configparser.ConfigParser:
        """
        Laad configuratie uit een INI bestand.

        Args:
            config_path: Pad naar het INI config bestand

        Returns:
            ConfigParser object met configuratie
        """
        config = configparser.ConfigParser()

        if not config.read(self.config_path):
            raise FileNotFoundError(f"Config bestand niet gevonden: {self.config_path}")

        return config

    def get(self, key: str, default: Any = None) -> Any:
        """Haal een config waarde op."""
        return self._config.get(key, default)

    def get_notifier_config(self, broker: str = "telegram") -> Dict[str, Any]:
        """Haal specifiek de notifier configuratie op."""
        return self._config.get(broker, {})

    def get_ipcameras_config(self) -> Dict[str, Any]:
        """Haal alle de ipcamera's configuratie op."""
        # TODO: add multiple camera's
        return self._config.get('ipcamera1', {})

    def get_ipcamera_config(self, json_top_level_key: str = "ipcamera1") -> Dict[str, Any]:
        """Haal specifieke ipcamera configuratie op."""
        return self._config.get(json_top_level_key, {})

    def get_detection_config(self, json_top_level_key: str = "detection_settings") -> Dict[str, Any]:
        """Haal generieke stuurparamters uit configuratie op."""
        return self._config.get(json_top_level_key, {})

    def get_all_ipcamera_keys(self) -> list[str]:
        """
        Haal alle ipcamera keys op.
        
        Returns:
            List met alle camera keys (bijv. ["ipcamera1", "ipcamera2", ...])
        """
        return [key for key in self._config.keys() if key.startswith("ipcamera")]
    
        


if __name__ == "__main__":
    logger = logging.getLogger("config.py - main")
    logger.info("\n=== Config Class ===")
    try:
        config = Config("config.json")

        #test notifier config
        notifier_config = config.get_notifier_config()
        # self.logger.info alle key value pairs uit de list:
        for key in notifier_config:
            logger.info(f"{key}: {notifier_config[key]} (type: {type(notifier_config[key]).__name__})")

        logger.info(f"Bot token: {notifier_config["api_key"]}")
        logger.info(f"Receipients: {notifier_config['chat_id']}")

        logger.info(f"Receipients is of type: {notifier_config['chat_id']}")

        # test ipcameras config
        ipcameras_config = config.get_ipcameras_config()
        # self.logger.info alle key value pairs uit de list:
        for key in ipcameras_config:
            logger.info(
                f"{key}: {ipcameras_config[key]} (type: {type(ipcameras_config[key]).__name__})"
            )

        # test ipcameras config
        ipcameras_config = config.get_ipcamera_config('ipcamera2')
        # self.logger.info alle key value pairs uit de list:
        for key in ipcameras_config:
            logger.info(
                f"{key}: {ipcameras_config[key]} (type: {type(ipcameras_config[key]).__name__})"
            )
    except Exception as e:
        logger.info(f"Fout: {e}")