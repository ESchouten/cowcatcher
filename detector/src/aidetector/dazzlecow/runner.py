import logging
from dataclasses import replace
from datetime import datetime

import numpy as np
from aidetector.dazzlecow.enrollment import export_named_gallery, finalize_enrollment
from aidetector.dazzlecow.gallery import DazzleCowGallery
from aidetector.dazzlecow.localizer import (
    CowCandidate,
    DazzleCowVideoLocalizer,
    LocalizerSettings,
)
from aidetector.dazzlecow.model import DazzleCowEncoder
from aidetector.dazzlecow.tracklet_store import TrackletStore
from aidetector.dazzlecow.tracks import TrackIdentityAggregator
from aidetector.utils.config import DazzleCowConfig, Detection, ImageSet
from numpy import ndarray

logger = logging.getLogger(__name__)


class DazzleCowRunner:
    def __init__(self, config: DazzleCowConfig):
        self.localizer = DazzleCowVideoLocalizer(
            LocalizerSettings.from_config(config),
            config.owl_interval,
        )
        self.encoder = DazzleCowEncoder(config.model, device=config.device)
        self.tracklet_store = (
            TrackletStore(config.enrollment.database)
            if config.enrollment is not None
            else None
        )
        self.gallery_path = (
            config.gallery
            if config.gallery is not None
            else config.enrollment.database.with_suffix(".npz")
            if config.enrollment is not None
            else None
        )
        self.gallery_settings = (
            config.neighbors,
            config.match_threshold,
            config.match_margin,
        )
        self.gallery = (
            DazzleCowGallery(
                config.gallery,
                neighbors=config.neighbors,
                match_threshold=config.match_threshold,
                match_margin=config.match_margin,
            )
            if config.gallery is not None and config.gallery.is_file()
            else None
        )
        self.enrollment_revision = -1
        self.enrollment_identity_count = (
            config.enrollment.identity_count if config.enrollment is not None else None
        )
        self._sync_enrollment()
        self.identity_tracks = TrackIdentityAggregator(
            self.gallery,
            samples=config.track_samples,
            iou_threshold=config.track_iou,
            max_age=config.track_max_age,
        )

    def detect(
        self,
        frames: list[ndarray],
        sources: list[str] | None = None,
        dates: list[datetime] | None = None,
    ) -> list[list[CowCandidate]]:
        sources = sources or [str(index) for index in range(len(frames))]
        dates = dates or [datetime.now()] * len(frames)
        if len(sources) != len(frames):
            raise ValueError("DazzleCow sources and frames differ in length")
        if len(dates) != len(frames):
            raise ValueError("DazzleCow dates and frames differ in length")
        results = []
        for source, frame, date in zip(sources, frames, dates, strict=True):
            candidates = self.localizer.locate(source, frame, date)
            embeddings = self.encoder.embed(
                [candidate.image for candidate in candidates]
            )
            snapshots = self.identity_tracks.apply(source, candidates, embeddings)
            tracklet_store = getattr(self, "tracklet_store", None)
            if tracklet_store is not None and not tracklet_store.is_finalized():
                for snapshot in snapshots:
                    tracklet_store.upsert(snapshot)
            self._sync_enrollment()
            results.append(candidates)
        return results

    def _sync_enrollment(self) -> None:
        store = getattr(self, "tracklet_store", None)
        if store is None:
            return
        if store.finalize_requested():
            try:
                if self.gallery_path is None:
                    raise ValueError("Enrollment has no gallery path")
                finalize_enrollment(
                    store,
                    self.gallery_path,
                    identity_count=getattr(self, "enrollment_identity_count", None),
                )
                logger.info("Finalized DazzleCow enrollment: %s", store.path)
            except ValueError as error:
                store.fail_finalize(str(error))
                logger.error("Could not finalize DazzleCow enrollment: %s", error)

        revision = store.revision()
        if revision == self.enrollment_revision or not store.is_finalized():
            return
        embeddings, identities = store.gallery_data()
        if self.gallery_path is not None:
            export_named_gallery(store, self.gallery_path)
        neighbors, match_threshold, match_margin = self.gallery_settings
        self.gallery = DazzleCowGallery.from_data(
            embeddings,
            np.asarray(identities),
            neighbors=neighbors,
            match_threshold=match_threshold,
            match_margin=match_margin,
        )
        if getattr(self, "identity_tracks", None) is not None:
            self.identity_tracks.gallery = self.gallery
        self.enrollment_revision = revision

    def detections_from_result(
        self,
        result: list[CowCandidate],
        frames: list[tuple[datetime, ndarray]],
    ) -> list[Detection] | None:
        if not result:
            return None

        crops = [replace(candidate.crop) for candidate in result]
        confidences = {
            "cow": max(crop.confidence or 0 for crop in crops),
        }
        detections = [
            Detection(date, ImageSet(frame, [replace(crop) for crop in crops]), {})
            for date, frame in frames[:-1]
        ]
        detections.append(
            Detection(frames[-1][0], ImageSet(frames[-1][1], crops), confidences)
        )
        return detections
