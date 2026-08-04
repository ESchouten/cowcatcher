import logging
from collections.abc import Iterator
from datetime import datetime
from threading import Event, Lock

from aidetector.adapters.sources.streaming import StreamBatcher
from aidetector.domain.frames import Frame, FrameBatch
from aidetector.utils.config import DetectionConfig, config_list
from ultralytics.data.loaders import LoadImagesAndVideos
from ultralytics.data.utils import IMG_FORMATS, VID_FORMATS

logger = logging.getLogger(__name__)


class SourceProvider:
    def __init__(self, config: DetectionConfig):
        self.sources = config_list(config.source)
        modes = {_is_realtime(source) for source in self.sources}
        if len(modes) != 1:
            raise ValueError("A detector cannot mix files and realtime sources")
        self.realtime = modes.pop()
        self.width = config.frames_width
        self.retention = config.frame_retention
        self._stop = Event()
        self._batcher: StreamBatcher | None = None
        self._batcher_lock = Lock()

    def iter_batches(self) -> Iterator[FrameBatch]:
        if self.realtime:
            yield from self._iter_stream_batches()
        else:
            yield from self._iter_file_batches()

    def _iter_stream_batches(
        self,
    ) -> Iterator[FrameBatch]:
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
    ) -> Iterator[FrameBatch]:
        loader = LoadImagesAndVideos(self.sources)
        try:
            for sources, images, _ in loader:
                if self._stop.is_set():
                    return
                yield {sources[0]: [Frame(datetime.now(), images[0])]}
        finally:
            if loader.cap is not None:
                loader.cap.release()

    def close(self) -> None:
        self._stop.set()
        with self._batcher_lock:
            batcher = self._batcher
        if batcher is not None:
            batcher.stop()


def _is_realtime(source: str) -> bool:
    is_file = source.lower().endswith(tuple(IMG_FORMATS.union(VID_FORMATS)))
    return source.isnumeric() or not is_file
