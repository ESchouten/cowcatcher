import logging
from collections.abc import Iterator
from datetime import datetime
from threading import Event, Lock

from aidetector.sources.streaming import StreamBatcher
from aidetector.utils.config import DetectionConfig
from numpy import ndarray
from ultralytics.data.loaders import LoadImagesAndVideos
from ultralytics.data.utils import IMG_FORMATS, VID_FORMATS

logger = logging.getLogger(__name__)


class SourceProvider:
    def __init__(self, config: DetectionConfig):
        self.sources = (
            [config.source] if isinstance(config.source, str) else list(config.source)
        )
        if not self.sources:
            raise ValueError("At least one detection source is required")
        self.width = config.frames_width
        self.retention = config.frame_retention
        self._stop = Event()
        self._batcher: StreamBatcher | None = None
        self._batcher_lock = Lock()

    def is_stream(self) -> bool:
        is_file = (
            self.sources[0].lower().endswith(tuple(IMG_FORMATS.union(VID_FORMATS)))
        )
        return self.sources[0].isnumeric() or not is_file

    def iter_batches(self) -> Iterator[dict[str, list[tuple[datetime, ndarray]]]]:
        if self.is_stream():
            yield from self._iter_stream_batches()
        else:
            yield from self._iter_file_batches()

    def _iter_stream_batches(
        self,
    ) -> Iterator[dict[str, list[tuple[datetime, ndarray]]]]:
        logger.info("Starting stream processing for %d source(s)", len(self.sources))
        batcher = StreamBatcher(self.sources, self.width, self.retention)
        with self._batcher_lock:
            self._batcher = batcher
        try:
            for batch in batcher:
                if self._stop.is_set():
                    return
                yield batch
        finally:
            batcher.stop()
            with self._batcher_lock:
                if self._batcher is batcher:
                    self._batcher = None

    def _iter_file_batches(
        self,
    ) -> Iterator[dict[str, list[tuple[datetime, ndarray]]]]:
        loader = LoadImagesAndVideos(self.sources)
        try:
            for sources, images, _ in loader:
                if self._stop.is_set():
                    return
                yield {sources[0]: [(datetime.now(), images[0])]}
        finally:
            if loader.cap is not None:
                loader.cap.release()

    def close(self) -> None:
        self._stop.set()
        with self._batcher_lock:
            batcher = self._batcher
        if batcher is not None:
            batcher.stop()
