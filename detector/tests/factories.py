from datetime import datetime

import numpy as np

from aidetector.domain.detections import DetectedObject, Observation
from aidetector.domain.events import DetectionEvent
from aidetector.domain.frames import Frame


def make_observation(
    captured_at: datetime | None = None,
    confidences: dict[str, float] | None = None,
    *,
    objects: tuple[DetectedObject, ...] = (),
    shape: tuple[int, int, int] = (80, 120, 3),
) -> Observation:
    return Observation(
        Frame(captured_at or datetime.now(), np.zeros(shape, dtype=np.uint8)),
        objects,
        {"cow": 0.9} if confidences is None else confidences,
    )


def make_event(
    observations: list[Observation] | None = None,
    *,
    source: str = "camera",
) -> DetectionEvent:
    items = observations or [make_observation()]
    return DetectionEvent(source, tuple(items), items[-1])
