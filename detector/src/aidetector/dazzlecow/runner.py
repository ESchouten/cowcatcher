import logging
from dataclasses import replace

from aidetector.adapters.models.yolo import YoloRunner
from aidetector.dazzlecow.enrollment import (
    finalize_enrollment,
    finalize_pending_enrollment,
)
from aidetector.dazzlecow.gallery import CowIdentityGallery, IdentityMatch
from aidetector.dazzlecow.localizer import (
    CowCandidate,
    DazzleCowLocalizer,
    LocalizerSettings,
    candidates_from_result,
)
from aidetector.dazzlecow.model import CowIdentityEncoder, IDENTITY_MODEL
from aidetector.dazzlecow.tracklet_store import TrackletStore
from aidetector.dazzlecow.tracks import TrackIdentityAggregator, TrackletSnapshot
from aidetector.domain.detections import Observation
from aidetector.domain.frames import Frame, FrameBatch
from aidetector.pipeline.ports import ModelBatchResult
from aidetector.utils.config import CowIdentityConfig, OnnxConfig, YoloConfig
from numpy import ndarray

logger = logging.getLogger(__name__)
MAX_LEARNED_SAMPLES_PER_IDENTITY = 20
MIN_LEARNING_SIMILARITY = 0.75
MIN_LEARNING_MARGIN = 0.1


class CowIdentityPipeline:
    def __init__(
        self,
        config: CowIdentityConfig,
        onnx_config: OnnxConfig,
        sources: list[str],
        yolo_config: YoloConfig,
        yolo_runner: YoloRunner,
    ):
        self.config = config
        self.reuses_primary_yolo = _can_reuse_primary_yolo(yolo_config, yolo_runner)
        self.localizer = (
            None
            if self.reuses_primary_yolo
            else DazzleCowLocalizer(
                LocalizerSettings.from_config(config),
                onnx_config,
                sources,
            )
        )
        self.encoder = CowIdentityEncoder()
        if config.enrollment is None and not config.database.is_file():
            raise FileNotFoundError(
                f"Cow identity database not found: {config.database}"
            )
        self.tracklet_store = TrackletStore(config.database)
        self.tracklet_store.ensure_embedding_model(
            IDENTITY_MODEL, self.encoder.feature_dim
        )
        self.database_revision = -1
        self.gallery: CowIdentityGallery | None = None
        self._sync_database()
        if config.enrollment is None and self.gallery is None:
            raise ValueError(
                f"Cow identity database is not finalized: {config.database}"
            )
        self.identity_tracks = TrackIdentityAggregator(
            self.gallery,
            samples=config.track_samples,
            max_age=config.track_max_age,
        )

    def track_sources(
        self,
        batch: FrameBatch,
    ) -> list[ModelBatchResult]:
        if self.localizer is None:
            raise RuntimeError("Primary YOLO supplies cow identity candidates")
        return [
            replace(
                tracked,
                result=self._identify(tracked.source, tracked.result),
            )
            for tracked in self.localizer.track_sources(batch)
        ]

    def candidates_from_primary(
        self,
        source: str,
        result,
        frame: ndarray,
    ) -> list[CowCandidate]:
        if not self.reuses_primary_yolo:
            raise RuntimeError("Cow identity uses its own segmentation model")
        candidates = candidates_from_result(
            result,
            frame,
            LocalizerSettings.from_config(self.config),
        )
        return self._identify(source, candidates)

    def live_observation(
        self,
        frame: Frame,
        candidates: list[CowCandidate],
    ) -> Observation:
        crops = [
            replace(candidate.crop, identities=tuple(candidate.crop.identities))
            for candidate in candidates
        ]
        return Observation(frame, tuple(crops))

    def close(self) -> None:
        self.tracklet_store.close()

    def _identify(
        self,
        source: str,
        candidates: list[CowCandidate],
    ) -> list[CowCandidate]:
        self._sync_database()
        embeddings = self.encoder.embed([candidate.image for candidate in candidates])
        snapshots = self.identity_tracks.apply(source, candidates, embeddings)
        if not self.tracklet_store.is_finalized():
            for snapshot in snapshots:
                self.tracklet_store.upsert(snapshot)
        else:
            for snapshot in snapshots:
                match = self._learning_match(snapshot)
                if match is not None:
                    try:
                        self.tracklet_store.update_identity(
                            snapshot,
                            max_samples=MAX_LEARNED_SAMPLES_PER_IDENTITY,
                        )
                        self.identity_tracks.mark_stored(
                            snapshot.source, snapshot.track_id
                        )
                    except ValueError as error:
                        logger.warning("Could not update cow identity: %s", error)
                elif snapshot.identity_key is None:
                    self.tracklet_store.update_pending(snapshot)
        self._sync_database()
        return candidates

    def _learning_match(self, snapshot: TrackletSnapshot) -> IdentityMatch | None:
        if (
            snapshot.identity_key is None
            or self.gallery is None
            or snapshot.observations < self.config.track_samples * 2
        ):
            return None
        match = self.gallery.match(snapshot.embedding)
        if (
            match is None
            or match.key != snapshot.identity_key
            or match.similarity
            < max(self.config.match_threshold, MIN_LEARNING_SIMILARITY)
            or match.margin < max(self.config.match_margin, MIN_LEARNING_MARGIN)
        ):
            return None
        return match

    def _sync_database(self) -> None:
        store = self.tracklet_store
        if store.finalize_requested():
            try:
                if store.is_finalized():
                    finalize_pending_enrollment(
                        store,
                        gallery=self.gallery,
                        existing_match_threshold=max(
                            self.config.match_threshold, MIN_LEARNING_SIMILARITY
                        ),
                        existing_match_margin=max(
                            self.config.match_margin, MIN_LEARNING_MARGIN
                        ),
                    )
                else:
                    finalize_enrollment(
                        store,
                        identity_count=(
                            self.config.enrollment.identity_count
                            if self.config.enrollment is not None
                            else None
                        ),
                    )
                logger.info("Finalized cow identity enrollment: %s", store.path)
            except ValueError as error:
                store.fail_finalize(str(error))
                logger.error("Could not finalize cow identity enrollment: %s", error)

        revision = store.revision()
        if revision == self.database_revision or not store.is_finalized():
            return
        embeddings, keys, labels = store.gallery_data()
        self.gallery = CowIdentityGallery(
            embeddings,
            keys,
            labels,
            match_threshold=self.config.match_threshold,
            match_margin=self.config.match_margin,
        )
        if getattr(self, "identity_tracks", None) is not None:
            self.identity_tracks.set_gallery(self.gallery)
        self.database_revision = revision


def _can_reuse_primary_yolo(config: YoloConfig, runner: YoloRunner) -> bool:
    return (
        config.task == "segment"
        and config.tracking
        and any(name == "cow" for name, _ in runner.class_confidences.values())
    )
