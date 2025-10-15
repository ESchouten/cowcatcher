from cowcatcher.config import Config, DetectorConfig
from cowcatcher.exporters.exporter import Exporter
from ultralytics.engine.results import Results
from datetime import datetime
import os
from pathlib import Path


class DiskExporter(Exporter):
    _directory: Path

    def __init__(self, config: Config, detector: DetectorConfig):
        super().__init__(config, detector)
        if detector.save_directory is None:
            raise ValueError("Save directory is not set for DiskExporter.")
        self._directory = detector.save_directory
        os.makedirs(self._directory, exist_ok=True)

    def export(self, data: Results):
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_name = f"detection_{now}.jpg"
        image_path = os.path.join(self._directory, image_name)
        data.save(image_path)  # pyright: ignore[reportUnknownMemberType]
