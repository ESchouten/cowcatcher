from datetime import datetime

from aidetector.media.video import shrink_image
from numpy import ndarray


class FrameCollector:
    frames: dict[str, list[tuple[datetime, ndarray]]]
    width: int
    retention: int

    def __init__(self, width: int = 1280, retention: int = 1):
        self.frames: dict[str, list[tuple[datetime, ndarray]]] = {}
        self.width = width
        self.retention = retention

    def add(self, source: str, frame: ndarray):
        frames = self.frames.setdefault(source, [])
        frames.append((datetime.now(), shrink_image(frame, self.width)))
        self.frames[source] = frames[-self.retention :]

    def clear(self):
        self.frames.clear()

    def counts(self) -> dict[str, int]:
        return {source: len(frames) for source, frames in self.frames.items()}

    def take(self) -> dict[str, list[tuple[datetime, ndarray]]]:
        snapshot = self.frames
        self.frames = {}
        return snapshot
