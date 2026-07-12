from __future__ import annotations

import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from threading import Thread
from time import sleep
from typing import TYPE_CHECKING

from aidetector.detection.identity_registry import IdentityRegistry
from aidetector.detection.validator import Validator
from aidetector.detection.yolo import YoloRunner
from aidetector.exporters.disk import DiskExporter
from aidetector.exporters.exporter import Exporter
from aidetector.exporters.sse import SSEExporter, default_sse_endpoint
from aidetector.exporters.telegram import TelegramExporter
from aidetector.exporters.webhook import WebhookExporter
from aidetector.sources.source import SourceProvider
from aidetector.utils.config import (
    ChatConfig,
    Config,
    CowIdentityConfig,
    Detection,
    DetectionConfig,
    DetectorConfig,
    DiskConfig,
    ImageSet,
    ModelConfig,
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

if TYPE_CHECKING:
    from aidetector.dazzlecow.runner import CowIdentityPipeline


class Detector:
    logger = logging.getLogger(__name__)
    detections: defaultdict[str, list[Detection]]
    detection: DetectionConfig
    yolo_config: YoloConfig | None
    yolo_runner: YoloRunner | None
    identity_config: CowIdentityConfig | None
    identity_pipeline: "CowIdentityPipeline | None"
    identity_registry: IdentityRegistry
    source_provider: SourceProvider
    validator: Validator
    exporters: list[Exporter]
    running: bool
    export_executor: ThreadPoolExecutor
    last_frame_time: datetime
    last_identity_result_time: dict[str, datetime]
    last_detection_time: dict[str, dict[str, datetime]]

    def __init__(
        self,
        detection: DetectionConfig,
        yolo_config: YoloConfig | None,
        identity_config: CowIdentityConfig | None,
        validator: Validator,
        exporters: list[Exporter],
        onnx_config: OnnxConfig,
        identity_registry: IdentityRegistry | None = None,
        identity_producer: str = "detector",
    ):
        self.detections = defaultdict(list)
        if identity_config is not None and yolo_config is None:
            raise ValueError("Cow identity requires a YOLO detector")

        self.detection = detection
        self.yolo_config = yolo_config
        self.identity_config = identity_config
        self.identity_registry = identity_registry or IdentityRegistry()
        self.identity_producer = identity_producer
        self.source_provider = SourceProvider(detection)
        self.yolo_runner = (
            YoloRunner(yolo_config, onnx_config, self.source_provider.sources)
            if yolo_config is not None
            else None
        )
        if identity_config is not None and self.yolo_runner is not None and yolo_config:
            from aidetector.dazzlecow.runner import CowIdentityPipeline

            self.identity_pipeline = CowIdentityPipeline(
                identity_config,
                onnx_config,
                self.source_provider.sources,
                yolo_config,
                self.yolo_runner,
            )
        else:
            self.identity_pipeline = None
        self.validator = validator
        self.exporters = exporters
        self.running = True
        self.export_executor = ThreadPoolExecutor()
        self.last_frame_time = datetime.min
        self.last_identity_result_time = {}
        self.last_detection_time = {}

    @classmethod
    def from_config(
        cls,
        config: Config,
        detector: DetectorConfig,
        detector_index: int = 0,
        identity_registry: IdentityRegistry | None = None,
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
                    if isinstance(item, SSEConfig) and item.endpoint is None:
                        item.endpoint = default_sse_endpoint(detector_index)
                    exporters.append(exporter_cls(item))

        validator = Validator.from_config(
            [detector.vlm]
            if isinstance(detector.vlm, VLMConfig)
            else detector.vlm or []
        )

        return [
            cls(
                detector.detection,
                detector.yolo,
                detector.identity,
                validator,
                exporters,
                config.onnx,
                identity_registry,
                f"detector-{detector_index}",
            )
        ]

    def _generate_frames(self):
        for batch in self.source_provider.iter_batches():
            if not self.running:
                return
            self._handle_frame_batch(batch)

    def _handle_frame_batch(self, batch: dict[str, list[tuple[datetime, ndarray]]]):
        identity_pipeline = getattr(self, "identity_pipeline", None)
        if identity_pipeline is not None and identity_pipeline.reuses_primary_yolo:
            self._handle_primary_identity_batch(batch, identity_pipeline)
            return

        if identity_pipeline is not None:
            for tracked in identity_pipeline.track_sources(batch):
                date, frame = tracked.frames[-1]
                identity_detection = identity_pipeline.live_detection(
                    date, frame, tracked.result
                )
                self.identity_registry.publish(
                    tracked.source,
                    self.identity_producer,
                    identity_detection,
                )
                self._publish_tracks(
                    tracked.source,
                    identity_detection,
                )

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

        runner = self.yolo_runner
        if runner is not None:
            if self.yolo_config and self.yolo_config.tracking and self.yolo_runner:
                tracked_results = self.yolo_runner.track_sources(batch)
                for tracked in tracked_results:
                    self._handle_model_result(
                        tracked.source,
                        tracked.result,
                        tracked.frames,
                    )
                return

            frames = [frames[-1][1] for frames in batch.values()]
            results = runner.detect(frames)
            for source, result in zip(batch.keys(), results):
                self._handle_model_result(source, result, batch[source])
            return

        for source, frames in batch.items():
            self._process(
                source,
                [Detection(frames[-1][0], ImageSet(frames[-1][1]), {})],
            )

    def _handle_primary_identity_batch(
        self,
        batch: dict[str, list[tuple[datetime, ndarray]]],
        identity_pipeline: CowIdentityPipeline,
    ) -> None:
        if self.yolo_runner is None:
            return
        result_times = self.last_identity_result_time
        for tracked in self.yolo_runner.track_sources(batch):
            source = tracked.source
            latest_date, latest_frame = tracked.frames[-1]
            candidates = identity_pipeline.candidates_from_primary(
                source,
                tracked.result,
                latest_frame,
            )
            identity_detection = identity_pipeline.live_detection(
                latest_date,
                latest_frame,
                candidates,
            )
            self.identity_registry.publish(
                source,
                self.identity_producer,
                identity_detection,
            )
            if (
                latest_date - result_times.get(source, datetime.min)
            ).total_seconds() >= self.detection.interval:
                result_times[source] = latest_date
                self._handle_model_result(
                    source,
                    tracked.result,
                    tracked.frames,
                )
                continue

            detections = self.yolo_runner.detections_from_result(
                tracked.result,
                [(latest_date, latest_frame)],
            )
            if detections:
                self.identity_registry.enrich(source, detections[-1])
            self._publish_tracks(
                source,
                detections[-1] if detections else identity_detection,
            )

    def _handle_model_result(
        self,
        source: str,
        result,
        frames: list[tuple[datetime, ndarray]],
    ):
        runner = self.yolo_runner
        model_config = self._model_config()
        if model_config is None or runner is None:
            return

        detections = runner.detections_from_result(result, frames)
        if detections:
            self.identity_registry.enrich(source, detections[-1])
            self._publish_tracks(source, detections[-1])
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
        if model_config.include_trailing_time > time_since_latest_detection:
            self.logger.info(
                "Including trailing frames: %f seconds", time_since_latest_detection
            )
            detections = [
                Detection(frame[0], ImageSet(frame[1]), {}) for frame in frames
            ]
            self._process(source, detections)

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

    def _export(self, source: str):
        detections = self.detections[source]
        if self._has_min_detections(source):
            best_detection = max(detections, key=lambda x: max_confidence(x.confidence))
            self.identity_registry.enrich(source, best_detection)

            model_config = self._model_config()
            matching_confs = (
                matching_confidences(best_detection.confidence, model_config.confidence)
                if model_config
                else []
            )
            if model_config and not self._cooldown_exceeded(source, matching_confs):
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

                if validated is not False and model_config:
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
            model_config.frames_min if (model_config := self._model_config()) else 0
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
        model_config = self._model_config()
        if model_config is None:
            return True

        def cooldown_for(name: str) -> float:
            return (
                model_config.cooldown[name]
                if isinstance(model_config.cooldown, dict)
                else model_config.cooldown
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
            model_config.time_max if (model_config := self._model_config()) else 0
        )
        return time_collecting_exceeded

    def _timeout_exceeded(self, source: str) -> bool:
        latest_detection = self._latest_detection(source)
        if not latest_detection:
            return False
        now = datetime.now()
        timeout = (now - latest_detection.date).total_seconds()
        return (
            timeout > model_config.timeout
            if (model_config := self._model_config()) and model_config.timeout
            else False
        )

    def _model_config(self) -> ModelConfig | None:
        return self.yolo_config
