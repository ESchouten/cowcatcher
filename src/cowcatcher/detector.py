from dataclasses import dataclass
from datetime import datetime

from ultralytics import YOLO  # pyright: ignore[reportPrivateImportUsage]
from ultralytics.engine.results import Results

from cowcatcher.config import DetectorConfig
from cowcatcher.exporters.exporter import Exporter


@dataclass
class Collecting:
    since: datetime
    max_confidence: float
    result: Results


class Detector:
    collecting: Collecting | None = None

    def __init__(self, config: DetectorConfig, exporters: list[Exporter]):
        self.config = config
        self.exporters = exporters
        print(f"Loading model from {config.model_url}")
        self.model = YOLO(config.model_url)
        self.detect()

    def detect(self):
        # ultralytics only supports one source for video files
        source = (
            self.config.sources[0] if self.config.sources[0].endswith((".mp4", ".avi", ".mov")) else self.config.sources
        )
        results = self.model.predict(  # pyright: ignore[reportUnknownMemberType]
            source,
            conf=self.config.confidence_threshold,
            project=self.config.save_directory,
            stream=True,
        )
        for result in results:
            self._try_set_collecting(result)
            self._try_export()

    def _try_set_collecting(self, result: Results):
        if result.boxes is not None and len(result.boxes) > 0:
            highest_confidence = max(box.conf.item() for box in result.boxes)  # pyright: ignore[reportUnknownMemberType]
            self.collecting = Collecting(
                since=self.collecting.since if self.collecting else datetime.now(),
                max_confidence=highest_confidence,
                result=result,
            )

    def _try_export(self):
        if (
            self.collecting is not None
            and (datetime.now() - self.collecting.since).total_seconds() >= self.config.collection_seconds
        ):
            print(f"Exporting detection with confidence {self.collecting.max_confidence}")
            for exporter in self.exporters:
                exporter.export(self.collecting.result)
            self.collecting = None
