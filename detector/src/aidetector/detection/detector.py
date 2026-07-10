import logging
from datetime import datetime
from threading import Event, Thread
from time import monotonic
from typing import Any, Protocol

from aidetector.detection.events import DetectionEvent, EventCollector
from aidetector.detection.models import Detection, ImageSet
from aidetector.exporters.dispatcher import ExportDispatcher
from aidetector.exporters.exporter import TrackPublisher
from aidetector.sources.source import SourceProvider
from aidetector.utils.config import DetectionConfig, YoloConfig
from numpy import ndarray

Frame = tuple[datetime, ndarray]
FrameBatch = dict[str, list[Frame]]


class ModelRunner(Protocol):
    def detect(self, frames: list[ndarray]) -> list[Any]: ...

    def track_sources(self, batch: FrameBatch) -> list[Any]: ...

    def detections_from_result(
        self,
        result: Any,
        frames: list[Frame],
    ) -> list[Detection] | None: ...


class Detector:
    logger = logging.getLogger(__name__)

    def __init__(
        self,
        detection: DetectionConfig,
        model_config: YoloConfig | None,
        source_provider: SourceProvider,
        model_runner: ModelRunner | None,
        event_collector: EventCollector | None,
        dispatcher: ExportDispatcher,
        track_publishers: list[TrackPublisher],
        detector_index: int = 0,
    ):
        if (model_config is None) != (model_runner is None):
            raise ValueError(
                "Model config and model runner must be configured together"
            )
        if model_config is not None and event_collector is None:
            raise ValueError("A model-backed detector requires an event collector")

        self.detection = detection
        self.model_config = model_config
        self.source_provider = source_provider
        self.model_runner = model_runner
        self.event_collector = event_collector
        self.dispatcher = dispatcher
        self.track_publishers = list(track_publishers)
        self._source_ids = {
            source: f"{detector_index}:{source_index}"
            for source_index, source in enumerate(source_provider.sources)
        }
        self.error: Exception | None = None
        self._stop = Event()
        self._producer: Thread | None = None
        self._monitor: Thread | None = None
        self._next_detection_at = 0.0
        self._started = False

    def start(self) -> Thread:
        if self._started:
            raise RuntimeError("Detector instances cannot be restarted")

        self._started = True
        self._stop.clear()
        self.error = None
        self._monitor = Thread(
            target=self._monitor_timeouts,
            name="detection-timeouts",
            daemon=True,
        )
        self._producer = Thread(
            target=self._run,
            name="detection-frames",
        )
        self._monitor.start()
        self._producer.start()
        return self._producer

    def stop(self) -> None:
        self._stop.set()
        self.source_provider.close()

    def close(self) -> None:
        self.stop()
        if self._started:
            self.join()
        else:
            self.dispatcher.close()

    def join(self, timeout: float | None = None) -> None:
        if self._producer is not None:
            self._producer.join(timeout)
        if self._monitor is not None:
            self._monitor.join(timeout)

    @property
    def is_alive(self) -> bool:
        return bool(self._producer and self._producer.is_alive())

    def _run(self) -> None:
        try:
            for batch in self.source_provider.iter_batches():
                if self._stop.is_set():
                    break
                self._handle_frame_batch(batch)
        except Exception as error:
            self.error = error
            self.logger.exception("Detector frame processing failed")
        finally:
            self._stop.set()
            self.source_provider.close()
            if self.event_collector is not None:
                self._submit(self.event_collector.flush_all())
            self.dispatcher.close()

    def _monitor_timeouts(self) -> None:
        while not self._stop.wait(1):
            if self.event_collector is not None:
                self._submit(self.event_collector.flush_expired())

    def _handle_frame_batch(self, batch: FrameBatch) -> None:
        if not batch or not self._detection_is_due():
            return

        if self.model_runner is None:
            for source, frames in batch.items():
                detections = [
                    Detection(date, ImageSet(frame), {}) for date, frame in frames
                ]
                if detections:
                    self.dispatcher.submit(
                        DetectionEvent(
                            self._source_ids[source],
                            tuple(detections),
                            detections[-1],
                        )
                    )
            return

        if self.model_config and self.model_config.tracking:
            for tracked in self.model_runner.track_sources(batch):
                self._handle_model_result(
                    tracked.source,
                    tracked.result,
                    tracked.frames,
                )
            return

        sources = list(batch)
        results = self.model_runner.detect([batch[source][-1][1] for source in sources])
        for source, result in zip(sources, results, strict=True):
            self._handle_model_result(source, result, batch[source])

    def _detection_is_due(self) -> bool:
        interval = max(0, self.detection.interval)
        now = monotonic()
        remaining = self._next_detection_at - now
        if remaining > 0:
            if self.source_provider.is_stream():
                return False
            if self._stop.wait(remaining):
                return False
            now = monotonic()
        self._next_detection_at = now + interval
        return True

    def _handle_model_result(
        self,
        source: str,
        result: Any,
        frames: list[Frame],
    ) -> None:
        assert self.model_runner is not None
        assert self.model_config is not None
        assert self.event_collector is not None

        detections = self.model_runner.detections_from_result(result, frames)
        source_id = self._source_ids[source]
        if detections:
            self._publish_tracks(source_id, detections[-1])
            self._submit(self.event_collector.add(source_id, detections))
            return

        latest_date, latest_frame = frames[-1]
        self._publish_tracks(
            source_id,
            Detection(latest_date, ImageSet(latest_frame), {}),
        )
        trailing = [Detection(date, ImageSet(frame), {}) for date, frame in frames]
        self._submit(self.event_collector.add_trailing(source_id, trailing))

    def _publish_tracks(self, source: str, detection: Detection) -> None:
        for publisher in self.track_publishers:
            try:
                publisher.publish_tracks(source, detection)
            except Exception:
                self.logger.exception(
                    "Track publisher %s failed",
                    publisher.__class__.__name__,
                )

    def _submit(self, events: list[DetectionEvent]) -> None:
        for event in events:
            self.dispatcher.submit(event)
