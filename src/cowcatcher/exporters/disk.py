import os
from datetime import datetime
from pathlib import Path
from typing import Self

from ultralytics.engine.results import Results

from cowcatcher.config import Config, DetectorConfig
from cowcatcher.exporters.exporter import Exporter


class DiskExporter(Exporter):
    save_directory: Path

    def __init__(self, save_directory: Path):
        super().__init__(save_directory)
        self.save_directory = os.path.join("detections", save_directory)
        os.makedirs(self.save_directory, exist_ok=True)

    @classmethod
    def fromConfig(cls, config: Config, detector: DetectorConfig) -> Self | None:
        if detector.save_directory is None:
            return None
        return cls(detector.save_directory)

    def export(self, data: Results):
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_name = f"detection_{now}.jpg"
        image_path = os.path.join(self.save_directory, image_name)
        data.save(image_path)
