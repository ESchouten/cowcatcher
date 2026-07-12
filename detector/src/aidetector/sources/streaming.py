import logging
from threading import Condition, Event, Lock, Thread

from ultralytics.data.loaders import LoadStreams

from aidetector.sources.collector import FrameCollector

logger = logging.getLogger(__name__)


class StreamBatcher:
    def __init__(self, sources: list[str], width: int = 1280, retention: int = 1):
        self.sources = list(sources)
        self.collector = FrameCollector(width, retention)
        self.loaders: list[LoadStreams | None] = [None] * len(sources)
        self.missing_sources: set[str] = set()
        self.condition = Condition()
        self._loaders_lock = Lock()
        self._stop = Event()
        self.threads = [
            Thread(
                target=self._run_loader,
                args=(index, source),
                name=f"stream-{index}",
                daemon=True,
            )
            for index, source in enumerate(self.sources)
        ]
        logger.info("Starting %d stream loader(s)", len(self.threads))
        for thread in self.threads:
            thread.start()

    def _run_loader(self, index: int, source: str) -> None:
        logger.info("Stream loader %d started", index)
        while not self._stop.is_set():
            loader: LoadStreams | None = None
            try:
                loader = LoadStreams(source)
                with self._loaders_lock:
                    self.loaders[index] = loader
                for _, images, _ in loader:
                    if self._stop.is_set():
                        break
                    if images is None:
                        continue
                    with self.condition:
                        self.collector.add(source, images[0])
                        logger.debug(
                            "Buffered %d frame(s) from source %d",
                            len(self.collector.frames[source]),
                            index,
                        )
                        self.condition.notify()
            except Exception:
                if not self._stop.is_set():
                    logger.exception("Stream loader %d crashed", index)
            finally:
                if loader is not None:
                    _close_loader(loader)
                with self._loaders_lock:
                    self.loaders[index] = None
            self._stop.wait(1)
        logger.info("Stream loader %d finished", index)

    def stop(self) -> None:
        if self._stop.is_set():
            return
        logger.info("Stopping %d stream loader(s)", len(self.sources))
        self._stop.set()
        with self.condition:
            self.condition.notify_all()
        with self._loaders_lock:
            loaders = list(self.loaders)
        for loader in loaders:
            if loader is not None:
                _close_loader(loader)
        for thread in self.threads:
            thread.join(timeout=5)
            if thread.is_alive():
                logger.warning("Stream loader %s did not stop in time", thread.name)

    def is_ready(self) -> bool:
        return len(self.collector.frames) == len(self.sources) or any(
            count >= 2 for count in self.collector.counts().values()
        )

    def _log_missing(self, present_sources: set[str]) -> None:
        missing = set(self.sources) - present_sources
        repeatedly_missing = missing & self.missing_sources
        if repeatedly_missing:
            indexes = [
                index
                for index, source in enumerate(self.sources)
                if source in repeatedly_missing
            ]
            logger.warning("Missing frames from source indexes: %s", indexes)
        self.missing_sources = missing

    def __iter__(self):
        while not self._stop.is_set():
            with self.condition:
                self.condition.wait_for(lambda: self._stop.is_set() or self.is_ready())
                if self._stop.is_set():
                    return
                snapshot = self.collector.take()
            if snapshot:
                self._log_missing(set(snapshot))
                yield snapshot


class _SuppressLoadStreamsFilter(logging.Filter):
    messages = (
        "Waiting for stream ",
        " (no detections), ",
        " postprocess per image at shape (",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        return not any(message in record.getMessage() for message in self.messages)


logging.getLogger("ultralytics").addFilter(_SuppressLoadStreamsFilter())


def _close_loader(loader: LoadStreams) -> None:
    try:
        loader.close()
    except Exception:
        logger.info("Failed to close stream loader", exc_info=True)
