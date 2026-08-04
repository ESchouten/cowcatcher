from datetime import datetime

from aidetector.domain.frames import Frame, FrameBatch
from aidetector.media.rendering import shrink_image
from numpy import ndarray


class FrameCollector:
    frames: FrameBatch
    width: int
    retention: int

    def __init__(self, width: int = 1280, retention: int = 1):
        self.frames: FrameBatch = {}
        self.width = width
        self.retention = retention

    def add(self, source: str, frame: ndarray):
        frames = self.frames.setdefault(source, [])
        frames.append(Frame(datetime.now(), shrink_image(frame, self.width)))
        self.frames[source] = frames[-self.retention :]

    def counts(self) -> dict[str, int]:
        return {source: len(frames) for source, frames in self.frames.items()}

    def take(self) -> FrameBatch:
        snapshot = self.frames
        self.frames = {}
        return snapshot
