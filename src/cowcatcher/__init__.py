import json
import logging

from cowcatcher.config import Config
from cowcatcher.detector import Detector
from cowcatcher.exporters.factory import create_exporters

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

config_json = json.load(open("config.json"))
if config_json is None:
    raise ValueError("Config file is empty or not found.")
config = Config(**config_json)

detectors = [Detector(detector, create_exporters(config, detector)) for detector in config.detectors]


def main():
    logger.info(f"Starting application with config: {config}")


if __name__ == "__main__":
    main()
