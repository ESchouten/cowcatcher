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
            completed = self._pop_expired(source, now)
            self._detections[source].extend(detections)
            if self._time_exceeded(source, now):
                completed.append(self._pop(source))
            return [event for event in completed if event is not None]

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
            completed = self._pop_expired(source, now)
            latest = self._latest_detection(source)
            if latest is None:
                return [event for event in completed if event is not None]
            elapsed = (detections[-1].date - latest.date).total_seconds()
            if elapsed <= self.config.include_trailing_time:
                self._detections[source].extend(detections)
            if self._time_exceeded(source, now):
                completed.append(self._pop(source))
            return [event for event in completed if event is not None]

    def flush_expired(self, *, now: datetime | None = None) -> list[DetectionEvent]:
        now = now or datetime.now()
        with self._lock:
            events = [
                event
                for source in list(self._detections)
                for event in self._pop_expired(source, now)
                if event is not None
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
    ) -> list[DetectionEvent | None]:
        if not self._timeout_exceeded(source, now):
            return []
        return [self._pop(source)]

    def _pop(self, source: str) -> DetectionEvent | None:
        detections = self._detections.pop(source, [])
        detected = [detection for detection in detections if detection.confidence]
        if len(detected) < self.config.frames_min:
            return None
        best = max(detected, key=lambda item: max_confidence(item.confidence))
        logger.info(
            "Collected %d frames for %s over %.1fs (confidence %.3f)",
            len(detections),
            source,
            (detections[-1].date - detections[0].date).total_seconds(),
            max_confidence(best.confidence),
        )
        return DetectionEvent(source, tuple(detections), best)

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
