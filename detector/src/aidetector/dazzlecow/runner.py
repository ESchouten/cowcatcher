from dataclasses import replace
from datetime import datetime

from aidetector.dazzlecow.gallery import DazzleCowGallery
from aidetector.dazzlecow.localizer import (
    CowCandidate,
    DazzleCowVideoLocalizer,
    LocalizerSettings,
)
from aidetector.dazzlecow.model import DazzleCowEncoder
from aidetector.dazzlecow.tracks import TrackIdentityAggregator
from aidetector.utils.config import DazzleCowConfig, Detection, ImageSet
from numpy import ndarray


class DazzleCowRunner:
    def __init__(self, config: DazzleCowConfig):
        self.localizer = DazzleCowVideoLocalizer(
            LocalizerSettings.from_config(config),
            config.owl_interval,
        )
        self.encoder = DazzleCowEncoder(config.model, device=config.device)
        self.gallery = DazzleCowGallery(
            config.gallery,
            neighbors=config.neighbors,
            match_threshold=config.match_threshold,
            match_margin=config.match_margin,
        )
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
            embeddings = self.encoder.embed([candidate.image for candidate in candidates])
            self.identity_tracks.apply(source, candidates, embeddings)
            results.append(candidates)
        return results

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
        detections.append(Detection(frames[-1][0], ImageSet(frames[-1][1], crops), confidences))
        return detections
