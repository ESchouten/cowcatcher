import logging
from threading import Event, Thread

import requests

from aidetector.utils.config import HealthcheckConfig

logger = logging.getLogger(__name__)


class Healthcheck:
    def __init__(self, config: HealthcheckConfig):
        self.config = config
        self._stop = Event()
        self._thread: Thread | None = None

    def start(self) -> None:
        self._thread = Thread(
            target=self._run,
            name="healthcheck",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    def _run(self) -> None:
        logger.info(
            "Starting healthcheck pinger (method=%s, interval=%ss, url=%s)",
            self.config.method,
            self.config.interval,
            self.config.url,
        )
        while not self._stop.is_set():
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
            self._stop.wait(self.config.interval)
