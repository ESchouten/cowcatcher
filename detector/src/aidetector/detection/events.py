import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from threading import Lock

from aidetector.detection.models import Detection, max_confidence
from aidetector.utils.config import YoloConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DetectionEvent:
    source: str
    detections: tuple[Detection, ...]
    best: Detection

    @property
    def started_at(self) -> datetime:
        return self.detections[0].date

    @property
    def ended_at(self) -> datetime:
        return self.detections[-1].date

    @property
    def duration(self) -> float:
        return (self.ended_at - self.started_at).total_seconds()

    @property
    def confidence(self) -> float:
        return max_confidence(self.best.confidence)


class EventCollector:
    """Builds independent detection events for each source."""

    def __init__(self, config: YoloConfig):
        self.config = config
        self._detections: defaultdict[str, list[Detection]] = defaultdict(list)
        self._lock = Lock()

    def add(
        self,
        source: str,
        detections: list[Detection],
        *,
        now: datetime | None = None,
    ) -> list[DetectionEvent]:
        now = now or datetime.now()
        with self._lock:
            return self._collect(source, detections, now, trailing=False)

    def add_trailing(
        self,
        source: str,
        detections: list[Detection],
        *,
        now: datetime | None = None,
    ) -> list[DetectionEvent]:
        if not detections:
            return []
        now = now or datetime.now()
        with self._lock:
            return self._collect(source, detections, now, trailing=True)

    def _collect(
        self,
        source: str,
        detections: list[Detection],
        now: datetime,
        *,
        trailing: bool,
    ) -> list[DetectionEvent]:
        expired = self._pop_expired(source, now)
        completed = [expired] if expired else []

        latest = self._latest_detection(source)
        if not trailing or (
            latest is not None
            and (detections[-1].date - latest.date).total_seconds()
            <= self.config.include_trailing_time
        ):
            self._detections[source].extend(detections)

        if self._time_exceeded(source, now) and (event := self._pop(source)):
            completed.append(event)
        return completed

    def flush_expired(self, *, now: datetime | None = None) -> list[DetectionEvent]:
        now = now or datetime.now()
        with self._lock:
            events = [
                event
                for source in list(self._detections)
                if (event := self._pop_expired(source, now)) is not None
            ]
        return events

    def flush_all(self) -> list[DetectionEvent]:
        with self._lock:
            events = [
                event
                for source in list(self._detections)
                if (event := self._pop(source)) is not None
            ]
        return events

    def _pop_expired(
        self,
        source: str,
        now: datetime,
    ) -> DetectionEvent | None:
        if not self._timeout_exceeded(source, now):
            return None
        return self._pop(source)

    def _pop(self, source: str) -> DetectionEvent | None:
        detections = self._detections.pop(source, [])
        detected = [detection for detection in detections if detection.confidence]
        if len(detected) < self.config.frames_min:
            return None
        best = max(detected, key=lambda item: max_confidence(item.confidence))
        event = DetectionEvent(source, tuple(detections), best)
        logger.info(
            "Collected %d frames for %s over %.1fs (confidence %.3f)",
            len(detections),
            source,
            event.duration,
            event.confidence,
        )
        return event

    def _latest_detection(self, source: str) -> Detection | None:
        return next(
            (
                detection
                for detection in reversed(self._detections.get(source, []))
                if detection.confidence
            ),
            None,
        )

    def _timeout_exceeded(self, source: str, now: datetime) -> bool:
        if not self.config.timeout:
            return False
        latest = self._latest_detection(source)
        return bool(
            latest and (now - latest.date).total_seconds() > self.config.timeout
        )

    def _time_exceeded(self, source: str, now: datetime) -> bool:
        detections = self._detections.get(source, [])
        return bool(
            detections
            and (now - detections[0].date).total_seconds() > self.config.time_max
        )
