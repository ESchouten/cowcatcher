import os
from pathlib import Path
from typing import Self

from aidetector.config import Config, Detection, DetectorConfig, get_date_path, get_timestamped_filename
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
        self.logger.info(f"Saving {len(sorted_detections)} photos to disk")
        timestamp = get_date_path(sorted_detections[0], "seconds")
        timestamped_directory = os.path.join(self.save_directory, timestamp)
        os.makedirs(timestamped_directory, exist_ok=True)
        for result in sorted_detections:
            image_name = get_timestamped_filename(result)
            image_path = os.path.join(timestamped_directory, image_name)
            with open(image_path, "wb") as f:
                f.write(result.jpg)
