import logging
from collections.abc import Callable
from dataclasses import dataclass
from threading import Event
from time import monotonic
from typing import Any

from aidetector.domain.detections import Observation
from aidetector.domain.events import DetectionEvent, LiveObservation
from aidetector.domain.frames import Frame, FrameBatch
from aidetector.pipeline.aggregation import EventAggregator
from aidetector.pipeline.dispatch import EventDispatcher
from aidetector.pipeline.ports import (
    EnrichmentBatch,
    FrameEnricher,
    ModelRunner,
    Sink,
)


@dataclass(frozen=True, slots=True)
class InferenceStage:
    tracking: bool
    runner: ModelRunner
    events: EventAggregator


class NoOpFrameEnricher:
    def process(self, _batch: FrameBatch) -> EnrichmentBatch:
        return EnrichmentBatch()

    def enrich(self, _source: str, observation: Observation) -> Observation:
        return observation


class FrameProcessor:
    logger = logging.getLogger(__name__)

    def __init__(
        self,
        *,
        interval: float,
        realtime: bool,
        source_ids: dict[str, str],
        inference: InferenceStage | None,
        enricher: FrameEnricher,
        dispatcher: EventDispatcher,
        live_sinks: list[Sink[LiveObservation]],
        compact: Callable[[Observation], Observation],
        stop: Event,
    ):
        self.interval = interval
        self.realtime = realtime
        self.source_ids = source_ids
        self.inference = inference
        self.enricher = enricher
        self.dispatcher = dispatcher
        self.live_sinks = list(live_sinks)
        self.compact = compact
        self.stop = stop
        self._next_detection_at = 0.0

    def process(self, batch: FrameBatch) -> None:
        if not batch:
            return
        enrichment = self.enricher.process(batch)
        if not self._detection_is_due():
            self._publish_enrichment(enrichment.observations)
            return

        if self.inference is None:
            for source, frames in batch.items():
                observations = tuple(
                    self.compact(Observation(frame)) for frame in frames
                )
                if observations:
                    self.dispatcher.submit(
                        DetectionEvent(
                            self.source_ids[source], observations, observations[-1]
                        )
                    )
            return

        if self.inference.tracking:
            tracked_results = enrichment.model_results
            if tracked_results is None:
                self._publish_enrichment(enrichment.observations)
                tracked_results = tuple(self.inference.runner.track_sources(batch))
            for tracked in tracked_results:
                self._handle_model_result(
                    tracked.source,
                    tracked.result,
                    tracked.frames,
                )
            return

        self._publish_enrichment(enrichment.observations)
        sources = list(batch)
        results = self.inference.runner.detect(
            [batch[source][-1].require_image() for source in sources]
        )
        for source, result in zip(sources, results, strict=True):
            self._handle_model_result(source, result, batch[source])

    def flush_expired(self) -> None:
        if self.inference is not None:
            self._submit(self.inference.events.flush_expired())

    def flush_all(self) -> None:
        if self.inference is not None:
            self._submit(self.inference.events.flush_all())

    def _detection_is_due(self) -> bool:
        now = monotonic()
        remaining = self._next_detection_at - now
        if remaining > 0:
            if self.realtime:
                return False
            if self.stop.wait(remaining):
                return False
            now = monotonic()
        self._next_detection_at = now + self.interval
        return True

    def _handle_model_result(
        self,
        source: str,
        result: Any,
        frames: list[Frame],
    ) -> None:
        assert self.inference is not None
        observations = self.inference.runner.observations_from_result(result, frames)
        source_id = self.source_ids[source]
        if observations:
            observations[-1] = self.enricher.enrich(source, observations[-1])
            self._publish(LiveObservation(source_id, observations[-1]))
            self._submit(self.inference.events.add(source_id, observations))
            return

        trailing = [Observation(frame) for frame in frames]
        self._publish(LiveObservation(source_id, trailing[-1]))
        self._submit(self.inference.events.add_trailing(source_id, trailing))

    def _publish_enrichment(
        self,
        observations: tuple[tuple[str, Observation], ...],
    ) -> None:
        for source, observation in observations:
            self._publish(LiveObservation(self.source_ids[source], observation))

    def _publish(self, message: LiveObservation) -> None:
        for sink in self.live_sinks:
            try:
                sink.send(message)
            except Exception:
                self.logger.exception("Live sink %s failed", sink.__class__.__name__)

    def _submit(self, events: list[DetectionEvent]) -> None:
        for event in events:
            self.dispatcher.submit(event)
