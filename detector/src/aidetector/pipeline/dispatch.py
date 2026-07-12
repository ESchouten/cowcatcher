import logging

from aidetector.domain.events import DetectionEvent
from aidetector.pipeline.cooldown import CooldownTracker
from aidetector.pipeline.messages import CompletedEvent
from aidetector.pipeline.ports import ArtifactFactory, EventValidator, Sink
from aidetector.pipeline.worker import LatestWinsWorker

logger = logging.getLogger(__name__)
DEFAULT_EVENT_QUEUE_CAPACITY = 2


class EventDispatcher:
    """Processes completed events serially through a bounded queue."""

    def __init__(
        self,
        validator: EventValidator,
        sinks: list[Sink[CompletedEvent]],
        cooldowns: CooldownTracker,
        artifact_factory: ArtifactFactory,
        *,
        capacity: int = DEFAULT_EVENT_QUEUE_CAPACITY,
    ):
        self.validator = validator
        self.sinks = list(sinks)
        self.cooldowns = cooldowns
        self.artifact_factory = artifact_factory
        self._worker = LatestWinsWorker(
            self._dispatch,
            name="detection-events",
            capacity=capacity,
            failure_message="Completed event processing failed",
        )

    def start(self) -> None:
        self._worker.start()

    def submit(self, event: DetectionEvent) -> None:
        if self._worker.submit(event):
            logger.warning("Dropped oldest event waiting for validation")

    def close(self) -> None:
        self._worker.close()

    def _dispatch(self, event: DetectionEvent) -> None:
        classes = self.cooldowns.eligible_classes(event)
        detected_at = event.best.frame.captured_at
        if classes == []:
            logger.info("Skipping %s during cooldown", event.source)
            return

        artifacts = self.artifact_factory(event)
        try:
            validated = self.validator.validate(event, artifacts)
        except Exception:
            logger.exception("Detection validation failed")
            return

        if validated is not False:
            self.cooldowns.record(event.source, classes or [], detected_at)

        message = CompletedEvent(event, validated, artifacts)
        for sink in self.sinks:
            try:
                sink.send(message)
            except Exception:
                logger.exception("Event sink %s failed", sink.__class__.__name__)
