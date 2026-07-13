import logging
from dataclasses import dataclass

from aidetector.reid.enrollment import (
    finalize_enrollment,
    finalize_pending_enrollment,
)
from aidetector.reid.gallery import IdentityGallery
from aidetector.reid.store import TrackletStore
from aidetector.domain.identity import IdentityMatch, TrackletSnapshot
from aidetector.pipeline.identity_provider import SamplingDecision

logger = logging.getLogger(__name__)

MAX_LEARNED_SAMPLES_PER_IDENTITY = 20
MIN_LEARNING_SIMILARITY = 0.75
MIN_LEARNING_MARGIN = 0.1


@dataclass(frozen=True, slots=True)
class CatalogPolicy:
    match_threshold: float
    match_margin: float
    track_samples: int
    enrollment_identity_count: int | None = None


class SqliteIdentityCatalog:
    def __init__(self, store: TrackletStore, policy: CatalogPolicy):
        self.store = store
        self.policy = policy
        self.gallery: IdentityGallery | None = None
        self._revision = -1

    def initialize(self, model: str, dimension: int) -> IdentityGallery | None:
        self.store.ensure_embedding_model(model, dimension)
        return self.sync()

    def sync(self) -> IdentityGallery | None:
        self._finalize_if_requested()
        revision = self.store.revision()
        if revision == self._revision or not self.store.is_finalized():
            return self.gallery

        embeddings, keys, labels = self.store.gallery_data()
        self.gallery = IdentityGallery(
            embeddings,
            keys,
            labels,
            match_threshold=self.policy.match_threshold,
            match_margin=self.policy.match_margin,
        )
        self._revision = revision
        return self.gallery

    def record(self, snapshot: TrackletSnapshot) -> SamplingDecision:
        if self.gallery is None:
            self.store.upsert(snapshot)
            return SamplingDecision.CONTINUE

        if self._learning_match(snapshot) is not None:
            try:
                self.store.update_identity(
                    snapshot,
                    max_samples=MAX_LEARNED_SAMPLES_PER_IDENTITY,
                )
            except ValueError as error:
                logger.warning("Could not update identity: %s", error)
                return SamplingDecision.CONTINUE
            return SamplingDecision.STOP

        if snapshot.identity_key is None:
            self.store.update_pending(snapshot)
        return SamplingDecision.CONTINUE

    def close(self) -> None:
        self.store.close()

    def _learning_match(self, snapshot: TrackletSnapshot) -> IdentityMatch | None:
        if (
            snapshot.identity_key is None
            or self.gallery is None
            or snapshot.observations < self.policy.track_samples * 2
        ):
            return None
        match = self.gallery.score(snapshot.embedding)
        if (
            match.key != snapshot.identity_key
            or match.similarity
            < max(self.policy.match_threshold, MIN_LEARNING_SIMILARITY)
            or match.margin < max(self.policy.match_margin, MIN_LEARNING_MARGIN)
        ):
            return None
        return match

    def _finalize_if_requested(self) -> None:
        if not self.store.finalize_requested():
            return
        try:
            if self.store.is_finalized():
                finalize_pending_enrollment(
                    self.store,
                    gallery=self.gallery,
                    existing_match_threshold=max(
                        self.policy.match_threshold,
                        MIN_LEARNING_SIMILARITY,
                    ),
                    existing_match_margin=max(
                        self.policy.match_margin,
                        MIN_LEARNING_MARGIN,
                    ),
                )
            else:
                finalize_enrollment(
                    self.store,
                    identity_count=self.policy.enrollment_identity_count,
                )
            logger.info("Finalized identity enrollment: %s", self.store.path)
        except ValueError as error:
            self.store.fail_finalize(str(error))
            logger.error("Could not finalize identity enrollment: %s", error)
