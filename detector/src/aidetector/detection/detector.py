import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from threading import Thread
from time import sleep
from typing import Any, cast

from aidetector.detection.validator import Validator
from aidetector.exporters.disk import DiskExporter
from aidetector.exporters.exporter import Exporter
from aidetector.exporters.telegram import TelegramExporter
from aidetector.exporters.webhook import WebhookExporter
from aidetector.sources.source import SourceProvider
from aidetector.utils.config import (
    ChatConfig,
    Config,
    Crop,
    Detection,
    DetectionConfig,
    DetectorConfig,
    DiskConfig,
    ImageSet,
    OnnxConfig,
    VLMConfig,
    WebhookConfig,
    YoloConfig,
    confidence_matches,
    matching_confidences,
    max_confidence,
    min_confidence,
)
from aidetector.utils.onnx import should_half, should_rect
from aidetector.utils.version import TYPE
from numpy import ndarray
from typing_extensions import Self
from ultralytics import YOLO


class Detector:
    logger = logging.getLogger(__name__)
    detections: defaultdict[str, list[Detection]]
    detection: DetectionConfig
    yolo: YOLO | None
    yolo_config: YoloConfig | None
    yolo_class_confidences: dict[int, tuple[str, float]]
    source_provider: SourceProvider
    validator: Validator
    exporters: list[Exporter]
    running: bool
    export_executor: ThreadPoolExecutor
    last_frame_time: datetime
    last_detection_time: dict[str, dict[str, datetime]]

    def __init__(
        self,
        detection: DetectionConfig,
        yolo_config: YoloConfig | None,
        validator: Validator,
        exporters: list[Exporter],
        onnx_config: OnnxConfig,
    ):
        self.detections = defaultdict(list)
        self.detection = detection
        self.yolo_config = yolo_config
        self.yolo = None
        self.yolo_class_confidences = {}
        self.source_provider = SourceProvider(detection)
        if yolo_config is not None:
            self.yolo = YOLO(
                yolo_config.model
                if yolo_config.model.endswith(".onnx") or TYPE == "cuda"
                else (
                    YOLO(yolo_config.model).export(
                        format="engine" if TYPE == "tensorrt" else "onnx",
                        batch=max(1, len(self.source_provider.sources)),
                        dynamic=True,
                        half=should_half(),
                        imgsz=yolo_config.imgsz,
                        simplify=True,
                        opset=onnx_config.opset,
                    )
                ),
                task="detect",
            )
            if self.yolo.predictor is None:
                self.yolo.predictor = self.yolo._smart_load("predictor")(
                    overrides=self.yolo.overrides,
                    _callbacks=self.yolo.callbacks,
                )
                self.yolo.predictor.setup_model(model=self.yolo.model, verbose=False)
            self.yolo_class_confidences = self._resolve_class_confidences(
                yolo_config.confidence
            )

        self.validator = validator
        self.exporters = exporters
        self.running = True
        self.export_executor = ThreadPoolExecutor()
        self.last_frame_time = datetime.min
        self.last_detection_time = {}

    @classmethod
    def from_config(cls, config: Config, detector: DetectorConfig) -> list[Self]:
        exporters: list[Exporter] = []
        if detector.exporters is not None:
            config_exporter_map = {
                "telegram": (ChatConfig, TelegramExporter),
                "webhook": (WebhookConfig, WebhookExporter),
                "disk": (DiskConfig, DiskExporter),
            }

            for config_name, (config_cls, exporter_cls) in config_exporter_map.items():
                config_obj = getattr(detector.exporters, config_name, []) or []
                config_list = (
                    [config_obj] if isinstance(config_obj, config_cls) else config_obj
                )
                for item in config_list:
                    exporters.append(
                        exporter_cls.from_config(config, detector, cast(Any, item))
                    )

        validator = Validator.from_config(
            [detector.vlm]
            if isinstance(detector.vlm, VLMConfig)
            else detector.vlm or []
        )

        return [
            cls(detector.detection, detector.yolo, validator, exporters, config.onnx)
        ]

    def _generate_frames(self):
        for batch in self.source_provider.iter_batches():
            if not self.running:
                return
            self._handle_frame_batch(batch)

    def _handle_frame_batch(self, batch: dict[str, list[tuple[datetime, ndarray]]]):
        if (
            datetime.now() - self.last_frame_time
        ).total_seconds() < self.detection.interval:
            sleep_for = max(
                0,
                self.detection.interval
                - (datetime.now() - self.last_frame_time).total_seconds(),
            )
            self.logger.info("Waiting for %f seconds before next detection", sleep_for)
            sleep(sleep_for)
            return
        self.last_frame_time = datetime.now()

        if self.yolo and self.yolo_config:
            if self.yolo_config.strategy == "LATEST":
                frames = [frames[-1][1] for frames in batch.values()]
            else:
                frames = [frame[1] for frames in batch.values() for frame in frames]

            then = datetime.now()
            results = self.yolo.predict(
                source=frames,
                conf=min_confidence(self.yolo_config.confidence),
                stream=False,
                classes=list(self.yolo_class_confidences.keys()) or None,
                imgsz=self.yolo_config.imgsz,
                rect=should_rect(),
                batch=len(frames),
            )
            now = datetime.now()
            self.logger.info(
                "Detection time: %dms for %d frame(s). Avg: %dms",
                (now - then).total_seconds() * 1000,
                len(frames),
                (now - then).total_seconds() * 1000 / len(frames),
            )
            if self.yolo_config.strategy == "LATEST":
                for source, result in zip(batch.keys(), results):
                    self._handle_yolo_result(source, result, batch[source])
            else:
                for source in batch.keys():
                    for i in range(len(batch[source])):
                        result = results.pop(0)
                        self._handle_yolo_result(
                            source, result, batch[source][i : i + 1]
                        )
            return

        for source, frames in batch.items():
            self._process(
                source,
                [Detection(frames[-1][0], ImageSet(frames[-1][1], None), {})],
            )

    def _handle_yolo_result(
        self, source: str, result, frames: list[tuple[datetime, ndarray]]
    ):
        if self.yolo_config is None:
            return

        confidences: dict[str, float] = {}
        for box in result.boxes:
            class_id = int(box.cls.item())
            class_name = self.yolo_class_confidences[class_id][0]
            confidence = box.conf.item()
            confidences[class_name] = max(confidences.get(class_name, 0), confidence)

        if not confidence_matches(confidences, self.yolo_config.confidence):
            self.logger.debug("Confidence does not match")
            latest_detection = self._latest_detection(source)
            if not latest_detection:
                return
            time_since_latest_detection = (
                (frames[-1][0] - latest_detection.date).total_seconds()
                if latest_detection
                else 0
            )
            if self.yolo_config.include_trailing_time > time_since_latest_detection:
                self.logger.info(
                    "Including trailing frames: %f seconds", time_since_latest_detection
                )
                detections = [
                    Detection(frame[0], ImageSet(frame[1], None), {})
                    for frame in frames
                ]
                self._process(source, detections)
            return

        best_box = max(result.boxes, key=lambda x: x.conf.item())
        x1, y1, x2, y2 = map(int, best_box.xyxy[0])
        class_id = int(best_box.cls.item())
        label = self.yolo_class_confidences[class_id][0]
        confidence = best_box.conf.item()

        detections = []
        for frame_data in frames[:-1]:
            detections.append(
                Detection(
                    frame_data[0],
                    ImageSet(
                        frame_data[1],
                        Crop(x1, y1, x2, y2, label=label, confidence=confidence),
                    ),
                    {},
                ),
            )
        detections.append(
            Detection(
                frames[-1][0],
                ImageSet(
                    frames[-1][1],
                    Crop(x1, y1, x2, y2, label=label, confidence=confidence),
                ),
                confidences,
            ),
        )
        self._process(source, detections)

    def start(self):
        def monitor_timeouts():
            self.logger.info("Starting timeout monitor")
            while self.running:
                self.logger.info("Checking for timeouts")
                try:
                    for source in list(self.detections.keys()):
                        self._process(source)
                except Exception:
                    self.logger.exception("Error in timeout monitor")
                sleep(1)

        def frame_producer():
            try:
                self._generate_frames()
            finally:
                self.running = False
                self.source_provider.close()
                self.export_executor.shutdown(wait=True)

        Thread(target=monitor_timeouts, daemon=True).start()
        thread = Thread(target=frame_producer)
        thread.start()
        return thread

    def _process(self, source: str, detections: list[Detection] | None = None):
        if self._timeout_exceeded(source):
            self._export(source)

        if detections:
            for detection in detections:
                self.detections[source].append(detection)

        if self._time_exceeded(source):
            self._export(source)

    def _resolve_class_confidences(
        self, confidence: float | dict[str, float]
    ) -> dict[int, tuple[str, float]]:
        if not self.yolo:
            return {}

        yolo_names = self.yolo.names
        id_to_name = (
            {
                int(class_id): str(class_name)
                for class_id, class_name in yolo_names.items()
            }
            if isinstance(yolo_names, dict)
            else {
                class_id: str(class_name)
                for class_id, class_name in enumerate(yolo_names)
            }
        )

        if not isinstance(confidence, dict):
            threshold = float(confidence)
            return {
                class_id: (class_name, threshold)
                for class_id, class_name in id_to_name.items()
            }

        name_to_id = {
            class_name: class_id for class_id, class_name in id_to_name.items()
        }
        class_confidences: dict[int, tuple[str, float]] = {}
        for raw_class_name, threshold in confidence.items():
            class_name = raw_class_name.strip()
            class_id = name_to_id.get(class_name)
            if class_id is None:
                available_names = ", ".join(
                    id_to_name[class_id] for class_id in sorted(id_to_name)
                )
                raise ValueError(
                    f"Unknown YOLO class name '{raw_class_name}' in yolo.confidence. "
                    f"Available class names: {available_names}"
                )
            class_confidences[class_id] = (id_to_name[class_id], float(threshold))

        return class_confidences

    def _export(self, source: str):
        detections = self.detections[source]
        if self._has_min_detections(source):
            best_detection = max(detections, key=lambda x: max_confidence(x.confidence))

            matching_confs = (
                matching_confidences(
                    best_detection.confidence, self.yolo_config.confidence
                )
                if self.yolo_config
                else []
            )
            if self.yolo_config and not self._cooldown_exceeded(source, matching_confs):
                self.logger.info(
                    "Not exporting, cooldown not exceeded for %s", matching_confs
                )
                self.detections[source] = []
                return

            self.logger.info(
                "Finished collecting with %s detections over %s seconds with max confidence %s",
                len(detections),
                (detections[-1].date - detections[0].date).total_seconds(),
                max_confidence(best_detection.confidence),
            )

            def export_task():
                validated = self.validator.validate(best_detection, detections)

                if validated is not False and self.yolo_config:
                    last_detection_time = self.last_detection_time.get(source, {})
                    for class_name in matching_confs:
                        last_detection_time[class_name] = best_detection.date
                    self.last_detection_time[source] = last_detection_time

                for exporter in self.exporters:
                    try:
                        exporter.export(best_detection, detections, validated)
                    except Exception:
                        self.logger.exception(
                            f"Exporter {exporter.__class__.__name__} failed"
                        )

            self.export_executor.submit(export_task)
        self.detections[source] = []

    def _has_min_detections(self, source: str) -> bool:
        detections_with_confidence = [
            detection for detection in self.detections[source] if detection.confidence
        ]
        return len(detections_with_confidence) >= (
            self.yolo_config.frames_min if self.yolo_config else 0
        )

    def _latest_detection(self, source: str) -> Detection | None:
        detections = self.detections[source]
        if not detections:
            return None
        detections_with_confidence = [
            detection for detection in detections if detection.confidence
        ]
        return detections_with_confidence[-1] if detections_with_confidence else None

    def _cooldown_exceeded(self, source: str, matching_confidences: list[str]) -> bool:
        yolo_config = self.yolo_config
        if yolo_config is None:
            return True

        def cooldown_for(name: str) -> float:
            return (
                yolo_config.cooldown[name]
                if isinstance(yolo_config.cooldown, dict)
                else yolo_config.cooldown
            )

        return any(
            datetime.now()
            - self.last_detection_time.get(source, {}).get(name, datetime.min)
            > timedelta(seconds=cooldown_for(name))
            for name in matching_confidences
        )

    def _time_exceeded(self, source: str) -> bool:
        detections = self.detections[source]
        if not detections:
            return False
        now = datetime.now()
        time_collecting = (now - detections[0].date).total_seconds()
        time_collecting_exceeded = time_collecting > (
            self.yolo_config.time_max if self.yolo_config else 0
        )
        return time_collecting_exceeded

    def _timeout_exceeded(self, source: str) -> bool:
        latest_detection = self._latest_detection(source)
        if not latest_detection:
            return False
        now = datetime.now()
        timeout = (now - latest_detection.date).total_seconds()
        return (
            timeout > self.yolo_config.timeout
            if self.yolo_config and self.yolo_config.timeout
            else False
        )
