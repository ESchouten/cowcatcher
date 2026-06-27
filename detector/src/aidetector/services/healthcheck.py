import logging
from threading import Thread
from time import sleep

import requests

from aidetector.utils.config import HealthcheckConfig

logger = logging.getLogger(__name__)


class Healthcheck:
    config: HealthcheckConfig
    running: bool

    def __init__(self, config: HealthcheckConfig):
        self.config = config
        self.running = True

    def start(self) -> Thread:
        thread = Thread(target=self._check, daemon=True)
        thread.start()
        return thread

    def stop(self) -> None:
        self.running = False

    def _check(self) -> None:
        logger.info(
            "Starting healthcheck pinger (method=%s, interval=%ss, url=%s)",
            self.config.method,
            self.config.interval,
            self.config.url,
        )
        while self.running:
            try:
                response = requests.request(
                    self.config.method,
                    self.config.url,
                    headers=self.config.headers,
                    data=self.config.body,
                    timeout=self.config.timeout,
                )
                if response.status_code >= 400:
                    logger.warning("Healthcheck ping returned %s", response.status_code)
            except requests.RequestException:
                logger.exception("Healthcheck ping failed")
            sleep(self.config.interval)
