import logging
from threading import Condition, Thread
from time import sleep

from ultralytics.data.loaders import LoadStreams

from .collector import FrameCollector

logger = logging.getLogger(__name__)


class StreamBatcher:
    running: bool
    threads: list[Thread]
    collector: FrameCollector
    loaders: list[LoadStreams | None]
    missing_sources: set[str]
    condition: Condition

    def __init__(self, sources: list[str], width: int = 1280, retention: int = 1):
        logger.info("Initializing StreamBatcher with %d sources", len(sources))
        self.running = True
        self.sources = sources
        self.loaders = [None] * len(sources)
        self.collector = FrameCollector(width, retention)
        self.threads = []
        self.missing_sources = set()
        self.condition = Condition()

        def run_loader(index: int, source: str):
            logger.info("Stream loader started for %s", source)
            while self.running:
                try:
                    loader = LoadStreams(source)
                    self.loaders[index] = loader
                    for _, imgs, _ in loader:
                        if not self.running:
                            break
                        if imgs is None:
                            continue
                        with self.condition:
                            self.collector.add(source, imgs[0])
                            logger.debug(
                                "Received frame %d from %s",
                                self.collector.frames[source],
                                source,
                            )
                            self.condition.notify()
                except Exception:
                    if self.running:
                        logger.exception("Stream loader crashed for %s", source)
                finally:
                    try:
                        loader.close()
                    except Exception:
                        logger.info("Failed to close stream loader", exc_info=True)
                sleep(1)
            logger.info("Stream loader finished for %s", source)

        for index, source in enumerate(self.sources):
            thread = Thread(target=run_loader, args=(index, source), daemon=True)
            thread.start()
            self.threads.append(thread)

    def stop(self) -> None:
        logger.info("Stopping StreamBatcher with %d active sources", len(self.sources))
        self.running = False
        with self.condition:
            self.condition.notify_all()
        for loader in self.loaders:
            try:
                if loader is not None:
                    loader.close()
            except Exception:
                logger.info("Failed to close stream loader", exc_info=True)
        self.loaders = []
        for thread in self.threads:
            thread.join()
        self.threads = []
        logger.info("StreamBatcher stopped")

    def is_ready(self) -> bool:
        return len(self.collector.frames) == len(self.sources) or any(
            count >= 2 for count in self.collector.counts().values()
        )

    def log_missing(self, present_sources: set[str]):
        new_missing = set(self.sources) - present_sources
        intersect = new_missing & self.missing_sources
        if intersect:
            logger.warning("Missing frames from sources: %s", sorted(intersect))
        self.missing_sources = new_missing

    def __iter__(self):
        logger.debug("StreamBatcher iterator started")
        while self.running:
            with self.condition:
                while not self.is_ready():
                    self.condition.wait()
                snapshot = dict(self.collector.frames)
                self.collector.clear()
            if snapshot:
                logger.debug(
                    "Yielding batch with %d frames from %d sources",
                    len(snapshot),
                    len(self.sources),
                )
                self.log_missing(set(snapshot.keys()))
                yield snapshot
        logger.debug("StreamBatcher iterator stopped")


ultralytics_logger = logging.getLogger("ultralytics")


class _SuppressLoadStreamsFilter(logging.Filter):
    filter_messages = [
        "Waiting for stream ",
        " (no detections), ",
        " postprocess per image at shape (",
    ]

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        if not message:
            return False
        return not any(part in message for part in self.filter_messages)


ultralytics_logger.addFilter(_SuppressLoadStreamsFilter())
