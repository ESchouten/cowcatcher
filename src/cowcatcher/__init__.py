import json
import logging

from cowcatcher.config import Config
from cowcatcher.detector import Detector
from cowcatcher.exporters.factory import create_exporters

logger = logging.getLogger(__name__)

config_json = json.load(open("test/config.json"))
config = Config(**config_json)

detectors = [Detector(detector, create_exporters(config, detector)) for detector in config.detectors]


def main():
    logger.info(f"Starting application with config: {config}")


if __name__ == "__main__":
    main()
