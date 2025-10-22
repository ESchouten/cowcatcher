import logging
import tempfile
from dataclasses import dataclass
from datetime import datetime
from threading import Thread
from typing import Self

import cv2
from ultralytics import YOLO
from ultralytics.engine.results import Results

from aidetector.config import CollectionConfig, Config, DetectorConfig
from aidetector.exporters.disk import DiskExporter
from aidetector.exporters.exporter import Exporter
from aidetector.exporters.telegram import TelegramExporter


@dataclass
class Collecting:
    since: datetime
    detections: list[datetime]
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
        return cls(detector.model_url, detector.sources, detector.collection, exporters)

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
            new_max_confidence = max(box.conf.item() for box in result.boxes)
            new_jpg = (
                cv2.imencode(".jpg", result.orig_img)[1]
                if self.collecting is None or new_max_confidence > self.collecting.max_confidence
                else None
            )
            detections = self.collecting.detections if self.collecting else []
            detections.append(datetime.now())

            self.collecting = Collecting(
                since=self.collecting.since if self.collecting else datetime.now(),
                detections=detections,
                max_confidence=new_max_confidence,
                jpg=new_jpg.tobytes() if new_jpg is not None else self.collecting.jpg,
            )

        if self.collecting is not None:
            self.collecting.detections = [
                d for d in detections if (datetime.now() - d).total_seconds() <= self.config.time_seconds
            ]

            if len(self.collecting.detections) == 0:
                self.collecting = None

    def _try_export(self):
        now: datetime = datetime.now()
        if self.collecting is None:
            return

        time_collecting = (now - self.collecting.since).total_seconds()
        if len(self.collecting.detections) < self.config.frames_min or time_collecting < self.config.time_seconds:
            return

        self.logger.info(
            f"Exporting collection with {len(self.collecting.detections)} detections over {time_collecting} seconds with max confidence {self.collecting.max_confidence}"
        )
        collecting = self.collecting

        def runner():
            for exporter in self.exporters:
                try:
                    exporter.export(collecting.jpg)
                except Exception:
                    self.logger.exception(f"Exporter {exporter.__class__.__name__} failed")

        Thread(target=runner, name=f"export-{now.isoformat()}", daemon=True).start()

        self.collecting = None
