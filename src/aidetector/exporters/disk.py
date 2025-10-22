import os
from pathlib import Path
from typing import Self

from aidetector.config import Config, Detection, DetectorConfig
from aidetector.exporters.exporter import Exporter


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

    def export(self, sorted_detections: list[Detection]):
        for result in sorted_detections:
            rounded_confidence = round(result.confidence, 3)
            date_str = result.date.isoformat(timespec="milliseconds").replace(":", "-")
            image_name = f"{date_str}_{rounded_confidence}.jpg"
            image_path = os.path.join(self.save_directory, image_name)
            with open(image_path, "wb") as f:
                f.write(result.jpg)
