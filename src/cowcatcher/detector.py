import logging
import tempfile
import threading
from dataclasses import dataclass
from datetime import datetime

from ultralytics import YOLO
from ultralytics.engine.results import Results

from cowcatcher.config import DetectorConfig
from cowcatcher.exporters.exporter import Exporter


@dataclass
class Collecting:
    since: datetime
    max_confidence: float
    result: Results


class Detector:
    logger = logging.getLogger(__name__)
    collecting: Collecting | None = None

    def __init__(self, config: DetectorConfig, exporters: list[Exporter]):
        self.config = config
        self.exporters = exporters
        self.logger.info(f"Loading model from {config.model_url}")
        self.model = YOLO(config.model_url)

        self.source = tempfile.mkstemp(suffix=".csv", text=True)[1]
        with open(self.source, "w", encoding="utf-8") as f:
            f.write(",".join(self.config.sources))

        self.detect()

    def detect(self):
        results = self.model.predict(
            source=self.source,
            conf=self.config.confidence_threshold,
            stream=True,
        )
        for result in results:
            self._try_set_collecting(result)
            self._try_export()

    def _try_set_collecting(self, result: Results):
        if result.boxes is not None and len(result.boxes) > 0:
            highest_confidence = max(box.conf.item() for box in result.boxes)
            if self.collecting is None or highest_confidence > self.collecting.max_confidence:
                self.collecting = Collecting(
                    since=self.collecting.since if self.collecting else datetime.now(),
                    max_confidence=highest_confidence,
                    result=result,
                )

    def _try_export(self):
        now: datetime = datetime.now()
        if (
            self.collecting is not None
            and (now - self.collecting.since).total_seconds() >= self.config.collection_seconds
        ):
            self.logger.info(f"Exporting detection with confidence {self.collecting.max_confidence}")
            collecting = self.collecting

            def runner():
                try:
                    for exporter in self.exporters:
                        exporter.export(collecting.result)
                except Exception:
                    self.logger.exception(f"Exporter {exporter.__class__.__name__} failed")

            t = threading.Thread(target=runner, name=f"export-{now.isoformat()}", daemon=True)
            t.start()

            self.collecting = None
