from datetime import datetime

from aidetector.media.video import shrink_image
from numpy import ndarray


class FrameCollector:
    frames: dict[str, list[tuple[datetime, ndarray]]]
    width: int
    retention: int

    def __init__(self, width: int = 1080, retention: int = 1):
        self.frames: dict[str, list[tuple[datetime, ndarray]]] = {}
        self.width = width
        self.retention = max(1, retention)

    def add(self, source: str, frame: ndarray):
        self.remove_old()
        if source not in self.frames:
            self.frames[source] = []

        self.frames[source].append((datetime.now(), shrink_image(frame, self.width)))

    def clear(self):
        self.frames.clear()

    def remove_old(self):
        for source, frames in self.frames.items():
            self.frames[source] = frames[-self.retention :]

    def counts(self) -> dict[str, int]:
        return {source: len(frames) for source, frames in self.frames.items()}
