import logging
import tempfile
from dataclasses import dataclass
from datetime import datetime
from threading import Thread
from typing import Self

import cv2
from ultralytics import YOLO
from ultralytics.engine.results import Results

from cowcatcher.config import Config, DetectorConfig
from cowcatcher.exporters.disk import DiskExporter
from cowcatcher.exporters.exporter import Exporter
from cowcatcher.exporters.telegram import TelegramExporter


@dataclass
class CollectionConfig:
    collection_seconds: int
    confidence_threshold: float


@dataclass
class Collecting:
    since: datetime
    max_confidence: float
    jpg: bytes


class Detector:
    logger = logging.getLogger(__name__)
    collecting: Collecting | None = None

    def __init__(self, model: str, sources: list[str], config: CollectionConfig, exporters: list[Exporter]):
        self.config = config
        self.exporters = exporters
        self.logger.info(f"Loading model from {model}")
        self.model = YOLO(model, task="detect")

        self.source = tempfile.mkstemp(suffix=".csv", text=True)[1]
        with open(self.source, "w", encoding="utf-8") as f:
            f.write(",".join(sources))

    @classmethod
    def fromConfig(cls, config: Config, detector: DetectorConfig) -> Self:
        exporterTypes: list[type[Self]] = [TelegramExporter, DiskExporter]
        exporters = list(filter(None, [exporter.fromConfig(config, detector) for exporter in exporterTypes]))
        return cls(
            detector.model_url,
            detector.sources,
            CollectionConfig(detector.collection_seconds, detector.confidence_threshold),
            exporters,
        )

    def start(self):
        def runner():
            results = self.model.predict(
                source=self.source,
                conf=self.config.confidence_threshold,
                stream=True,
            )
            for result in results:
                self._try_set_collecting(result)
                self._try_export()

        Thread(target=runner).start()

    def _try_set_collecting(self, result: Results):
        if result.boxes is not None and len(result.boxes) > 0:
            highest_confidence = max(box.conf.item() for box in result.boxes)
            if self.collecting is None or highest_confidence > self.collecting.max_confidence:
                success, jpg = cv2.imencode(".jpg", result.orig_img)
                if success:
                    self.collecting = Collecting(
                        since=self.collecting.since if self.collecting else datetime.now(),
                        max_confidence=highest_confidence,
                        jpg=jpg.tobytes(),
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
                for exporter in self.exporters:
                    try:
                        exporter.export(collecting.jpg)
                    except Exception:
                        self.logger.exception(f"Exporter {exporter.__class__.__name__} failed")

            Thread(target=runner, name=f"export-{now.isoformat()}", daemon=True).start()

            self.collecting = None
