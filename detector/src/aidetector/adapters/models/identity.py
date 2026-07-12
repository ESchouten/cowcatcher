from typing import TYPE_CHECKING

from aidetector.domain.detections import Observation
from aidetector.domain.frames import FrameBatch
from aidetector.pipeline.identity import IdentityRegistry
from aidetector.pipeline.ports import EnrichmentBatch, ModelBatchResult, ModelRunner

if TYPE_CHECKING:
    from aidetector.dazzlecow.runner import CowIdentityPipeline


class IdentityEnricher:
    def __init__(
        self,
        registry: IdentityRegistry,
        producer: str,
        pipeline: "CowIdentityPipeline | None",
        primary_runner: ModelRunner | None,
    ):
        self.registry = registry
        self.producer = producer
        self.pipeline = pipeline
        self.primary_runner = primary_runner

    def start(self) -> None:
        pass

    def close(self) -> None:
        if self.pipeline is not None:
            self.pipeline.close()

    def process(self, batch: FrameBatch) -> EnrichmentBatch:
        pipeline = self.pipeline
        if pipeline is None:
            return EnrichmentBatch()

        if pipeline.reuses_primary_yolo:
            assert self.primary_runner is not None
            tracked = self.primary_runner.track_sources(batch)
        else:
            tracked = pipeline.track_sources(batch)

        observations = []
        for item in tracked:
            frame = item.frames[-1]
            candidates = (
                pipeline.candidates_from_primary(
                    item.source,
                    item.result,
                    frame.require_image(),
                )
                if pipeline.reuses_primary_yolo
                else item.result
            )
            identity_observation = pipeline.live_observation(frame, candidates)
            self.registry.publish(item.source, self.producer, identity_observation)
            observations.append(
                (
                    item.source,
                    self._primary_observation(item) or identity_observation,
                )
            )

        return EnrichmentBatch(
            tuple(observations),
            tuple(tracked) if pipeline.reuses_primary_yolo else None,
        )

    def enrich(self, source: str, observation: Observation) -> Observation:
        return self.registry.enrich(source, observation)

    def _primary_observation(self, item: ModelBatchResult) -> Observation | None:
        if self.pipeline is None or not self.pipeline.reuses_primary_yolo:
            return None
        assert self.primary_runner is not None
        observations = self.primary_runner.observations_from_result(
            item.result,
            [item.frames[-1]],
        )
        return (
            self.registry.enrich(item.source, observations[-1])
            if observations
            else None
        )
