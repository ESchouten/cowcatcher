import logging
from collections.abc import Callable
from threading import Event, Lock, Thread

from aidetector.domain.detections import Observation
from aidetector.domain.events import LiveObservation
from aidetector.pipeline.dispatch import EventDispatcher
from aidetector.pipeline.inference import (
    FrameProcessor,
    InferenceStage,
    NoOpFrameEnricher,
)
from aidetector.pipeline.ports import FrameEnricher, FrameSource, Resource, Sink


class DetectionPipeline:
    logger = logging.getLogger(__name__)

    def __init__(
        self,
        *,
        interval: float,
        source: FrameSource,
        inference: InferenceStage | None,
        enricher: FrameEnricher | None = None,
        dispatcher: EventDispatcher,
        live_sinks: list[Sink[LiveObservation]],
        compact: Callable[[Observation], Observation],
        resources: list[Resource],
        pipeline_index: int = 0,
    ):
        self.source = source
        self.dispatcher = dispatcher
        self.resources = list(resources)
        self.error: Exception | None = None
        self._stop = Event()
        self._producer: Thread | None = None
        self._monitor: Thread | None = None
        self._finished = False
        self._finish_lock = Lock()
        source_ids = {
            raw_source: f"{pipeline_index}:{source_index}"
            for source_index, raw_source in enumerate(source.sources)
        }
        self.processor = FrameProcessor(
            interval=interval,
            realtime=source.realtime,
            source_ids=source_ids,
            inference=inference,
            enricher=enricher or NoOpFrameEnricher(),
            dispatcher=dispatcher,
            live_sinks=live_sinks,
            compact=compact,
            stop=self._stop,
        )

    def start(self) -> None:
        for resource in self.resources:
            resource.start()
        self.dispatcher.start()
        if self.processor.inference is not None:
            self._monitor = Thread(
                target=self._monitor_timeouts,
                name="detection-timeouts",
                daemon=True,
            )
            self._monitor.start()
        self._producer = Thread(target=self._run, name="detection-frames")
        self._producer.start()

    def stop(self) -> None:
        self._stop.set()
        self.source.close()

    def close(self) -> None:
        self.stop()
        if self._producer is not None:
            self.join()
        else:
            self._finish()

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
            for batch in self.source.iter_batches():
                if self._stop.is_set():
                    break
                self.processor.process(batch)
        except Exception as error:
            self.error = error
            self.logger.exception("Detection pipeline failed")
        finally:
            self.stop()
            self._finish()

    def _finish(self) -> None:
        with self._finish_lock:
            if self._finished:
                return
            self._finished = True
        self.processor.flush_all()
        self.dispatcher.close()
        for resource in self.resources:
            try:
                resource.close()
            except Exception:
                self.logger.exception("Failed to close %s", resource.__class__.__name__)

    def _monitor_timeouts(self) -> None:
        while not self._stop.wait(1):
            self.processor.flush_expired()
