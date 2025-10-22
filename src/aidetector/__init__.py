import logging

from aidetector.config import config
from aidetector.manager import Manager

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info(f"Starting application with config: {config}")
    manager = Manager.fromConfig(config)
    manager.start()


if __name__ == "__main__":
    main()
