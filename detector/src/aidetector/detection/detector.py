import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from threading import Thread
from time import sleep

from aidetector.detection.validator import Validator
from aidetector.detection.yolo import YoloRunner
from aidetector.exporters.disk import DiskExporter
from aidetector.exporters.exporter import Exporter
from aidetector.exporters.sse import SSEExporter
from aidetector.exporters.telegram import TelegramExporter
from aidetector.exporters.webhook import WebhookExporter
from aidetector.identity.enricher import IdentityEnricher
from aidetector.identity.provider import IdentityProvider
from aidetector.sources.source import SourceProvider
from aidetector.utils.config import (
    ChatConfig,
    Config,
    Detection,
    DetectionConfig,
    DetectorIdentityConfig,
    DetectorConfig,
    DiskConfig,
    ImageSet,
    OnnxConfig,
    SSEConfig,
    VLMConfig,
    WebhookConfig,
    YoloConfig,
    matching_confidences,
    max_confidence,
)
from numpy import ndarray
from typing_extensions import Self


class Detector:
    logger = logging.getLogger(__name__)
    detections: defaultdict[str, list[Detection]]
    detection: DetectionConfig
    yolo_config: YoloConfig | None
    yolo_runner: YoloRunner | None
    source_provider: SourceProvider
    validator: Validator
    exporters: list[Exporter]
    identity_enricher: IdentityEnricher | None
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
        identity_provider: IdentityProvider | None = None,
        identity_config: DetectorIdentityConfig | None = None,
    ):
        self.detections = defaultdict(list)
        self.detection = detection
        self.yolo_config = yolo_config
        self.source_provider = SourceProvider(detection)
        self.yolo_runner = (
            YoloRunner(yolo_config, onnx_config, len(self.source_provider.sources))
            if yolo_config is not None
            else None
        )
        self.validator = validator
        self.exporters = exporters
        self.identity_enricher = (
            IdentityEnricher(identity_provider, identity_config)
            if identity_provider is not None and identity_config is not None
            else None
        )
        self.running = True
        self.export_executor = ThreadPoolExecutor()
        self.last_frame_time = datetime.min
        self.last_detection_time = {}

    @classmethod
    def from_config(
        cls,
        config: Config,
        detector: DetectorConfig,
        identity_providers: dict[str, IdentityProvider] | None = None,
    ) -> list[Self]:
        exporters: list[Exporter] = []
        if detector.exporters is not None:
            config_exporter_map = {
                "telegram": (ChatConfig, TelegramExporter),
                "webhook": (WebhookConfig, WebhookExporter),
                "disk": (DiskConfig, DiskExporter),
                "sse": (SSEConfig, SSEExporter),
            }

            for config_name, (config_cls, exporter_cls) in config_exporter_map.items():
                config_obj = getattr(detector.exporters, config_name, []) or []
                config_list = (
                    [config_obj] if isinstance(config_obj, config_cls) else config_obj
                )
                for item in config_list:
                    exporters.append(exporter_cls(item))

        validator = Validator.from_config(
            [detector.vlm]
            if isinstance(detector.vlm, VLMConfig)
            else detector.vlm or []
        )
        identity_provider = None
        if detector.identity is not None:
            identity_provider = (identity_providers or {}).get(detector.identity.provider)
            if identity_provider is None:
                raise ValueError(
                    f"Unknown identity provider id: {detector.identity.provider}"
                )

        return [
            cls(
                detector.detection,
                detector.yolo,
                validator,
                exporters,
                config.onnx,
                identity_provider,
                detector.identity,
            )
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

        if self.yolo_runner and self.yolo_config:
            if self.yolo_config.tracking:
                for source, source_frames in batch.items():
                    if self.yolo_config.strategy == "LATEST":
                        frames = [source_frames[-1][1]]
                        results = self.yolo_runner.predict(frames, source=source)
                        for result in results:
                            self._handle_yolo_result(source, result, source_frames)
                    else:
                        frames = [frame[1] for frame in source_frames]
                        results = self.yolo_runner.predict(frames, source=source)
                        for i, result in enumerate(results):
                            self._handle_yolo_result(
                                source, result, source_frames[i : i + 1]
                            )
                return

            if self.yolo_config.strategy == "LATEST":
                frames = [frames[-1][1] for frames in batch.values()]
            else:
                frames = [frame[1] for frames in batch.values() for frame in frames]

            results = self.yolo_runner.predict(frames)
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
                [Detection(frames[-1][0], ImageSet(frames[-1][1]), {})],
            )

    def _handle_yolo_result(
        self, source: str, result, frames: list[tuple[datetime, ndarray]]
    ):
        if self.yolo_config is None or self.yolo_runner is None:
            return

        detections = self.yolo_runner.detections_from_result(result, frames)
        if detections:
            latest_detection = detections[-1]
            if self.identity_enricher:
                self.identity_enricher.enrich_live(source, latest_detection)
            self._publish_tracks(source, latest_detection)
            self._process(source, detections)
            return

        self.logger.debug("Confidence does not match")
        self._publish_tracks(
            source,
            Detection(frames[-1][0], ImageSet(frames[-1][1]), {}),
        )
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
                Detection(frame[0], ImageSet(frame[1]), {})
                for frame in frames
            ]
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

    def _publish_tracks(self, source: str, detection: Detection) -> None:
        for exporter in self.exporters:
            publish_tracks = getattr(exporter, "publish_tracks", None)
            if publish_tracks is None:
                continue
            try:
                publish_tracks(source, detection)
            except Exception:
                self.logger.exception(
                    "Exporter %s failed to publish tracks",
                    exporter.__class__.__name__,
                )

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
                if validated is not False and self.identity_enricher:
                    try:
                        self.identity_enricher.enrich(source, best_detection, detections)
                        self._publish_tracks(source, best_detection)
                    except Exception:
                        self.logger.exception("Identity lookup failed")

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
