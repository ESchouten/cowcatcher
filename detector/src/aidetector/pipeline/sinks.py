import logging

from aidetector.domain.policies import ExportPolicy
from aidetector.pipeline.messages import CompletedEvent
from aidetector.pipeline.ports import Sink
from aidetector.pipeline.worker import LatestWinsWorker

logger = logging.getLogger(__name__)
SINK_QUEUE_CAPACITY = 2


class FilteredEventSink:
    def __init__(self, policy: ExportPolicy, target: Sink[CompletedEvent]):
        self.policy = policy
        self.target = target
        self.logger = logging.getLogger(target.__class__.__name__)

    def send(self, message: CompletedEvent) -> None:
        if not self.policy.accepts(message.event, message.validated):
            self.logger.info("Event does not satisfy export policy")
            return
        self.target.send(message)


class BufferedSink:
    def __init__(self, target: Sink[CompletedEvent]):
        self.target = target
        name = f"sink-{target.__class__.__name__.lower()}"
        self._worker = LatestWinsWorker(
            target.send,
            name=name,
            capacity=SINK_QUEUE_CAPACITY,
            failure_message=f"Sink {target.__class__.__name__} failed",
        )

    def start(self) -> None:
        self._worker.start()

    def send(self, message: CompletedEvent) -> None:
        if self._worker.submit(message):
            logger.warning(
                "Dropped oldest event for slow %s", self.target.__class__.__name__
            )

    def close(self) -> None:
        self._worker.close()
