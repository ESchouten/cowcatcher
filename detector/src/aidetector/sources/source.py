import logging
from collections.abc import Iterator
from datetime import datetime

from aidetector.sources.streaming import StreamBatcher
from aidetector.utils.config import DetectionConfig
from numpy import ndarray
from ultralytics.data.loaders import LoadImagesAndVideos
from ultralytics.data.utils import IMG_FORMATS, VID_FORMATS

logger = logging.getLogger(__name__)


class SourceProvider:
    running: bool
    sources: list[str]
    width: int
    retention: int

    def __init__(self, detection: DetectionConfig):
        self.running = True
        self.sources = (
            [detection.source]
            if isinstance(detection.source, str)
            else detection.source
        )
        self.width = detection.frames_width
        self.retention = detection.frame_retention

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
        logger.info("Starting stream processing for sources: %s", self.sources)
        batcher = StreamBatcher(self.sources, self.width, self.retention)
        logger.info(
            "StreamBatcher started with %d active sources", len(batcher.sources)
        )

        try:
            for batch in batcher:
                if not self.running:
                    return
                yield batch
            logger.info("StreamBatcher loop ended")
        finally:
            batcher.stop()

    def _iter_file_batches(self) -> Iterator[dict[str, list[tuple[datetime, ndarray]]]]:
        results = LoadImagesAndVideos(self.sources)
        for sources, imgs, _ in results:
            if not self.running:
                return
            yield {sources[0]: [(datetime.now(), imgs[0])]}

    def close(self):
        self.running = False
