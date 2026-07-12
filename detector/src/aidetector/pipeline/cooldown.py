from datetime import datetime, timedelta

from aidetector.domain.detections import matching_confidences
from aidetector.domain.events import DetectionEvent
from aidetector.domain.policies import CooldownPolicy


class CooldownTracker:
    def __init__(self, policy: CooldownPolicy | None):
        self.policy = policy
        self._last_detection: dict[str, dict[str, datetime]] = {}

    def eligible_classes(self, event: DetectionEvent) -> list[str] | None:
        if self.policy is None:
            return None
        previous = self._last_detection.get(event.source, {})
        detected_at = event.best.frame.captured_at
        return [
            class_name
            for class_name in matching_confidences(
                event.best.confidences,
                self.policy.confidence,
            )
            if detected_at - previous.get(class_name, datetime.min)
            > timedelta(seconds=self.policy.for_class(class_name))
        ]

    def record(
        self,
        source: str,
        classes: list[str],
        detected_at: datetime,
    ) -> None:
        previous = self._last_detection.setdefault(source, {})
        for class_name in classes:
            previous[class_name] = detected_at
