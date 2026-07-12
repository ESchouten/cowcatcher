import logging
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime
from threading import Lock

from aidetector.domain.detections import Observation, max_confidence
from aidetector.domain.events import DetectionEvent
from aidetector.domain.policies import EventPolicy

logger = logging.getLogger(__name__)


class EventAggregator:
    """Builds independent detection events for each source."""

    def __init__(
        self,
        policy: EventPolicy,
        compact: Callable[[Observation], Observation] | None = None,
    ):
        self.policy = policy
        self.compact = compact or (lambda observation: observation)
        self._observations: defaultdict[str, list[Observation]] = defaultdict(list)
        self._lock = Lock()

    def add(
        self,
        source: str,
        observations: list[Observation],
        *,
        now: datetime | None = None,
    ) -> list[DetectionEvent]:
        now = now or datetime.now()
        with self._lock:
            return self._collect(source, observations, now, trailing=False)

    def add_trailing(
        self,
        source: str,
        observations: list[Observation],
        *,
        now: datetime | None = None,
    ) -> list[DetectionEvent]:
        if not observations:
            return []
        now = now or datetime.now()
        with self._lock:
            return self._collect(source, observations, now, trailing=True)

    def _collect(
        self,
        source: str,
        observations: list[Observation],
        now: datetime,
        *,
        trailing: bool,
    ) -> list[DetectionEvent]:
        expired = self._pop_expired(source, now)
        completed = [expired] if expired else []

        latest = self._latest_detection(source)
        if not trailing or (
            latest is not None
            and (
                observations[-1].frame.captured_at - latest.frame.captured_at
            ).total_seconds()
            <= self.policy.trailing_time
        ):
            self._observations[source].extend(map(self.compact, observations))

        if self._time_exceeded(source, now) and (event := self._pop(source)):
            completed.append(event)
        return completed

    def flush_expired(self, *, now: datetime | None = None) -> list[DetectionEvent]:
        now = now or datetime.now()
        with self._lock:
            return [
                event
                for source in list(self._observations)
                if (event := self._pop_expired(source, now)) is not None
            ]

    def flush_all(self) -> list[DetectionEvent]:
        with self._lock:
            return [
                event
                for source in list(self._observations)
                if (event := self._pop(source)) is not None
            ]

    def _pop_expired(
        self,
        source: str,
        now: datetime,
    ) -> DetectionEvent | None:
        return self._pop(source) if self._timeout_exceeded(source, now) else None

    def _pop(self, source: str) -> DetectionEvent | None:
        observations = self._observations.pop(source, [])
        detected = [item for item in observations if item.confidences]
        if len(detected) < self.policy.frames_min:
            return None
        best = max(detected, key=lambda item: max_confidence(item.confidences))
        event = DetectionEvent(source, tuple(observations), best)
        logger.info(
            "Collected %d frames for %s over %.1fs (confidence %.3f)",
            len(observations),
            source,
            event.duration,
            event.confidence,
        )
        return event

    def _latest_detection(self, source: str) -> Observation | None:
        return next(
            (
                item
                for item in reversed(self._observations.get(source, []))
                if item.confidences
            ),
            None,
        )

    def _timeout_exceeded(self, source: str, now: datetime) -> bool:
        if not self.policy.timeout:
            return False
        latest = self._latest_detection(source)
        return bool(
            latest
            and (now - latest.frame.captured_at).total_seconds() > self.policy.timeout
        )

    def _time_exceeded(self, source: str, now: datetime) -> bool:
        observations = self._observations.get(source, [])
        return bool(
            observations
            and (now - observations[0].frame.captured_at).total_seconds()
            > self.policy.time_max
        )
