import logging
import tempfile
from datetime import datetime
from threading import Thread
from typing import Self

import cv2
from ultralytics import YOLO
from ultralytics.engine.results import Results

from aidetector.config import CollectionConfig, Config, Detection, DetectorConfig
from aidetector.exporters.disk import DiskExporter
from aidetector.exporters.exporter import Exporter
from aidetector.exporters.telegram import TelegramExporter


class Detector:
    logger = logging.getLogger(__name__)
    detections: list[Detection] = []

    def __init__(
        self,
        model: str,
        sources: list[str],
        config: CollectionConfig,
        exporters: list[Exporter],
    ):
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
        exporters = list(
            filter(
                None,
                [exporter.fromConfig(config, detector) for exporter in exporterTypes],
            )
        )
        return cls(detector.model_url, detector.sources, detector.collection, exporters)

    def start(self):
        def runner():
            results = self.model.predict(
                source=self.source,
                conf=self.config.confidence_threshold,
                stream=True,
            )
            for result in results:
                self._add_detection(result)
                self._try_export()
                self._filter_detections()

        Thread(target=runner).start()

    def _filter_detections(self):
        self.detections = [
            d
            for d in self.detections
            if (datetime.now() - d.date).total_seconds() <= self.config.time_seconds
        ]

    def _add_detection(self, result: Results):
        if result.boxes is not None and len(result.boxes) > 0:
            confidence = max(box.conf.item() for box in result.boxes)
            success, jpg = cv2.imencode(".jpg", result.orig_img)
            if not success:
                return

            self.detections.append(
                Detection(date=datetime.now(), jpg=jpg.tobytes(), confidence=confidence)
            )

    def _try_export(self):
        now: datetime = datetime.now()
        if not self.detections:
            return

        time_collecting = (now - self.detections[0].date).total_seconds()
        if (
            len(self.detections) < self.config.frames_min
            or time_collecting < self.config.time_seconds
        ):
            return

        self.logger.info(
            f"Exporting collection with {len(self.detections)} detections over {time_collecting} seconds with max confidence {max(d.confidence for d in self.detections)}"
        )
        sorted_detections = sorted(
            self.detections, key=lambda d: d.confidence, reverse=True
        )

        def runner():
            for exporter in self.exporters:
                try:
                    exporter.export(sorted_detections)
                except Exception:
                    self.logger.exception(
                        f"Exporter {exporter.__class__.__name__} failed"
                    )

        Thread(target=runner, daemon=True).start()

        self.detections = []
