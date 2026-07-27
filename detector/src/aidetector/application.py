from __future__ import annotations

from time import sleep
from typing import TYPE_CHECKING

from aidetector.adapters.models.yolo import (
    YoloRunner,
    build_yolo_model,
)
from aidetector.adapters.sinks.factory import build_sinks
from aidetector.adapters.sources.source import SourceProvider
from aidetector.adapters.validation.vlm import VLMValidator
from aidetector.domain.policies import CooldownPolicy, EventPolicy
from aidetector.media.artifacts import EventArtifacts
from aidetector.media.storage import compact_observation
from aidetector.pipeline.aggregation import EventAggregator
from aidetector.pipeline.cooldown import CooldownTracker
from aidetector.pipeline.dispatch import EventDispatcher
from aidetector.pipeline.inference import InferenceStage
from aidetector.pipeline.processor import DetectionPipeline
from aidetector.services.healthcheck import Healthcheck
from aidetector.utils.config import (
    Config,
    DetectorConfig,
    IdentityConfig,
    OnnxConfig,
    config_list,
)

if TYPE_CHECKING:
    from aidetector.reid.stage import IdentityStage


class Application:
    def __init__(
        self,
        pipelines: list[DetectionPipeline],
        health: Healthcheck | None,
    ):
        self.pipelines = pipelines
        self.health = health

    @classmethod
    def from_config(cls, config: Config) -> "Application":
        pipelines = [
            build_pipeline(
                detector_config,
                config.onnx,
                index,
            )
            for index, detector_config in enumerate(config.detectors)
        ]
        health = Healthcheck(config.health) if config.health else None
        return cls(pipelines, health)

    def start(self) -> None:
        try:
            for pipeline in self.pipelines:
                pipeline.start()
            if self.health:
                self.health.start()
        except Exception:
            self.stop()
            raise

    def wait(self) -> None:
        while True:
            failed = next(
                (pipeline for pipeline in self.pipelines if pipeline.error is not None),
                None,
            )
            if failed is not None:
                raise RuntimeError("Detection pipeline stopped unexpectedly") from (
                    failed.error
                )
            if not any(pipeline.is_alive for pipeline in self.pipelines):
                return
            sleep(0.1)

    def stop(self) -> None:
        for pipeline in self.pipelines:
            pipeline.stop()
        if self.health:
            self.health.stop()
        for pipeline in self.pipelines:
            pipeline.close()


def build_pipeline(
    config: DetectorConfig,
    onnx: OnnxConfig,
    pipeline_index: int,
) -> DetectionPipeline:
    source = SourceProvider(config.detection)
    inference = None
    runner = None
    cooldown_policy = None
    if config.yolo is not None:
        yolo = config.yolo
        model = build_yolo_model(yolo, onnx, len(source.sources))
        runner = YoloRunner(yolo, source.sources, model)
        inference = InferenceStage(
            tracking=yolo.tracking,
            runner=runner,
            events=EventAggregator(
                EventPolicy(
                    frames_min=yolo.frames_min,
                    timeout=yolo.timeout,
                    time_max=yolo.time_max,
                    trailing_time=yolo.include_trailing_time,
                ),
                compact_observation,
            ),
        )
        cooldown_policy = CooldownPolicy(yolo.confidence, yolo.cooldown)

    identity_stage = None
    if config.identity is not None:
        assert config.yolo is not None and runner is not None
        identity_stage = build_identity_stage(
            config.identity,
            runner,
        )

    sinks = build_sinks(config.exporters, pipeline_index)
    dispatcher = EventDispatcher(
        VLMValidator(config_list(config.vlm)),
        sinks.events,
        CooldownTracker(cooldown_policy),
        EventArtifacts,
    )
    return DetectionPipeline(
        interval=config.detection.interval,
        source=source,
        inference=inference,
        identity_stage=identity_stage,
        dispatcher=dispatcher,
        live_sinks=sinks.live,
        compact=compact_observation,
        resources=[
            *sinks.resources,
            *([identity_stage] if identity_stage is not None else []),
        ],
        pipeline_index=pipeline_index,
    )


def build_identity_stage(
    config: IdentityConfig,
    primary_runner: YoloRunner,
) -> IdentityStage:
    from aidetector.reid.identity_catalog import IdentityCatalog
    from aidetector.reid.miewid import build_miewid_encoder
    from aidetector.reid.stage import IdentityStage

    available_labels = {
        name for name, _threshold in primary_runner.class_confidences.values()
    }
    if config.target_label not in available_labels:
        raise ValueError(
            f"Identity target label '{config.target_label}' is not tracked "
            "by the selected detector"
        )

    return IdentityStage(
        target_label=config.target_label,
        zone=(
            config.zone.x1,
            config.zone.y1,
            config.zone.x2,
            config.zone.y2,
        ),
        encoder=build_miewid_encoder(),
        catalog=IdentityCatalog(config.database),
    )
