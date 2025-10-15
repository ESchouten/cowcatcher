from abc import ABC, abstractmethod

from ultralytics.engine.results import Results

from cowcatcher.config import Config, DetectorConfig


class Exporter(ABC):
    def __init__(self, config: Config, detector: DetectorConfig):
        self.config = config
        self.detector = detector

    @abstractmethod
    def export(self, data: Results):
        pass
