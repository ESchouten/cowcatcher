import argparse
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from aidetector.dazzlecow.gallery import DazzleCowGallery
from aidetector.dazzlecow.geometry import box_iou
from aidetector.dazzlecow.localizer import (
    CowCandidate,
    DazzleCowLocalizer,
    DazzleCowVideoLocalizer,
    LocalizerSettings,
    segment_candidates,
)
from aidetector.dazzlecow.model import DazzleCowEncoder
from aidetector.dazzlecow.tracks import TrackIdentityAggregator
from numpy import ndarray

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GroundTruth:
    identity: str
    box: tuple[int, int, int, int]


@dataclass(frozen=True)
class VideoObservation:
    frame_id: int
    ground_truth: list[GroundTruth]
    candidates: list[CowCandidate]
    embeddings: ndarray


@dataclass
class _Metrics:
    ground_truth: int = 0
    predictions: int = 0
    localized: int = 0
    identified: int = 0
    correct: int = 0
    track_id_switches: int = 0
    identity_switches: int = 0
    track_first_seen: dict[int, int] = field(default_factory=dict)
    track_identified_after: dict[int, int] = field(default_factory=dict)
    tracks_by_identity: defaultdict[str, set[int]] = field(
        default_factory=lambda: defaultdict(set)
    )
    last_track_by_identity: dict[str, int] = field(default_factory=dict)
    last_result_by_identity: dict[str, str] = field(default_factory=dict)

    def add(
        self,
        observation_index: int,
        observation: VideoObservation,
        candidates: list[CowCandidate],
        matches: list[tuple[int, GroundTruth]],
    ) -> None:
        self.ground_truth += len(observation.ground_truth)
        self.predictions += len(candidates)
        self.localized += len(matches)
        for candidate_index, truth in matches:
            crop = candidates[candidate_index].crop
            if crop.track_id is None:
                continue
            track_id = crop.track_id
            self.track_first_seen.setdefault(track_id, observation_index)
            self.tracks_by_identity[truth.identity].add(track_id)
            previous_track = self.last_track_by_identity.get(truth.identity)
            if previous_track is not None and previous_track != track_id:
                self.track_id_switches += 1
            self.last_track_by_identity[truth.identity] = track_id

            if crop.identity is None:
                continue
            self.identified += 1
            result = crop.identity.identity
            self.track_identified_after.setdefault(
                track_id,
                observation_index - self.track_first_seen[track_id],
            )
            previous_result = self.last_result_by_identity.get(truth.identity)
            if previous_result is not None and previous_result != result:
                self.identity_switches += 1
            self.last_result_by_identity[truth.identity] = result
            self.correct += result == truth.identity

    def report(self) -> dict[str, float | int]:
        track_count = len(self.track_first_seen)
        fragments = [len(tracks) for tracks in self.tracks_by_identity.values()]
        delays = list(self.track_identified_after.values())
        return {
            "ground_truth_objects": self.ground_truth,
            "predictions": self.predictions,
            "localized_objects": self.localized,
            "localization_recall": self.localized / self.ground_truth
            if self.ground_truth
            else 0,
            "localization_precision": self.localized / self.predictions
            if self.predictions
            else 0,
            "identity_coverage": self.identified / self.localized
            if self.localized
            else 0,
            "identity_accuracy": self.correct / self.identified
            if self.identified
            else 0,
            "end_to_end_accuracy": self.correct / self.ground_truth
            if self.ground_truth
            else 0,
            "track_count": track_count,
            "track_id_switches": self.track_id_switches,
            "identity_switches": self.identity_switches,
            "extra_track_fragments": sum(max(0, count - 1) for count in fragments),
            "mean_tracks_per_identity": float(np.mean(fragments)) if fragments else 0,
            "tracks_without_identity": track_count - len(self.track_identified_after),
            "mean_observations_to_identity": float(np.mean(delays)) if delays else 0,
        }


def evaluate_observations(
    observations: Iterable[VideoObservation],
    gallery: DazzleCowGallery,
    *,
    sample_counts: list[int],
    track_iou: float,
    track_max_age: int,
    ground_truth_iou: float,
) -> dict[str, dict[str, float | int]]:
    trackers = {
        samples: TrackIdentityAggregator(
            gallery,
            samples=samples,
            iou_threshold=track_iou,
            max_age=track_max_age,
        )
        for samples in sorted(set(sample_counts))
    }
    metrics = {samples: _Metrics() for samples in trackers}
    for observation_index, observation in enumerate(observations):
        for samples, tracker in trackers.items():
            candidates = [
                CowCandidate(
                    replace(candidate.crop, track_id=None, identity=None),
                    candidate.image,
                )
                for candidate in observation.candidates
            ]
            tracker.apply("video", candidates, observation.embeddings)
            matches = match_ground_truth(
                candidates,
                observation.ground_truth,
                ground_truth_iou,
            )
            metrics[samples].add(
                observation_index,
                observation,
                candidates,
                matches,
            )
    return {str(samples): value.report() for samples, value in metrics.items()}


def match_ground_truth(
    candidates: list[CowCandidate],
    ground_truth: list[GroundTruth],
    threshold: float,
) -> list[tuple[int, GroundTruth]]:
    pairs = sorted(
        (
            (box_iou(_candidate_box(candidate), truth.box), index, truth)
            for index, candidate in enumerate(candidates)
            for truth in ground_truth
        ),
        key=lambda item: item[0],
        reverse=True,
    )
    matches = []
    used_candidates = set()
    used_identities = set()
    for score, index, truth in pairs:
        if score < threshold:
            break
        if index in used_candidates or truth.identity in used_identities:
            continue
        matches.append((index, truth))
        used_candidates.add(index)
        used_identities.add(truth.identity)
    return matches


def video_observations(
    video: Path,
    annotations: Path,
    localize,
    encoder: DazzleCowEncoder,
    *,
    start_frame: int,
    end_frame: int | None,
    frame_step: int,
    max_frames: int | None,
) -> Iterable[VideoObservation]:
    import pandas as pd

    table = pd.read_pickle(annotations).set_index("frame_id")
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 1
    stop = min(end_frame or frame_count, frame_count)
    selected = range(max(1, start_frame), stop + 1, max(1, frame_step))
    try:
        for index, frame_id in enumerate(selected):
            if max_frames is not None and index >= max_frames:
                break
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id - 1)
            success, frame = capture.read()
            if not success:
                logger.warning("Could not read frame %d", frame_id)
                continue
            ground_truth = _ground_truth(table.loc[frame_id], width, height)
            candidates = localize(
                frame,
                ground_truth,
                datetime.fromtimestamp(frame_id / fps),
            )
            embeddings = encoder.embed([candidate.image for candidate in candidates])
            logger.info(
                "Frame %d: %d/%d candidates",
                frame_id,
                len(candidates),
                len(ground_truth),
            )
            yield VideoObservation(frame_id, ground_truth, candidates, embeddings)
    finally:
        capture.release()


def _ground_truth(rows, width: int, height: int) -> list[GroundTruth]:
    if getattr(rows, "ndim", 1) == 1:
        rows = rows.to_frame().T
    return [
        GroundTruth(
            f"{int(row.tracklet_id):03d}",
            (
                round((row.x - row.w / 2) * width),
                round((row.y - row.h / 2) * height),
                round((row.x + row.w / 2) * width),
                round((row.y + row.h / 2) * height),
            ),
        )
        for row in rows.itertuples()
    ]


def _candidate_box(candidate: CowCandidate) -> tuple[int, int, int, int]:
    crop = candidate.crop
    return crop.x1, crop.y1, crop.x2, crop.y2
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark the complete DazzleCow video identity pipeline"
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--gallery", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--localizer",
        choices=("video", "image", "annotations"),
        default="video",
    )
    parser.add_argument("--owl-model", default="google/owlv2-large-patch14-ensemble")
    parser.add_argument("--sam-model", default="sam2.1_l.pt")
    parser.add_argument("--owl-interval", type=float, default=1)
    parser.add_argument("--prompt", default="cow")
    parser.add_argument("--confidence", type=float, default=0.3)
    parser.add_argument("--min-area-ratio", type=float, default=0.025)
    parser.add_argument("--max-area-ratio", type=float, default=0.075)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--neighbors", type=int, default=5)
    parser.add_argument("--match-threshold", type=float, default=0.75)
    parser.add_argument("--match-margin", type=float, default=0)
    parser.add_argument("--track-samples", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--track-iou", type=float, default=0.2)
    parser.add_argument("--track-max-age", type=int, default=10)
    parser.add_argument("--ground-truth-iou", type=float, default=0.3)
    parser.add_argument("--start-frame", type=int, default=1)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument("--frame-step", type=int, default=20)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--device", default="auto")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    settings = LocalizerSettings(
        arguments.owl_model,
        arguments.sam_model,
        arguments.prompt,
        arguments.confidence,
        arguments.min_area_ratio,
        arguments.max_area_ratio,
        arguments.nms_iou,
        arguments.device,
    )
    if arguments.localizer == "video":
        localizer = DazzleCowVideoLocalizer(settings, arguments.owl_interval)

        def localize(frame, _truth, date):
            return localizer.locate("video", frame, date)

    elif arguments.localizer == "image":
        localizer = DazzleCowLocalizer(
            settings
        )

        def localize(frame, _truth, _date):
            return localizer.locate(frame)

    else:
        from ultralytics import SAM

        sam = SAM(arguments.sam_model)
        def localize(frame, truth, _date):
            return segment_candidates(
                sam,
                frame,
                [list(item.box) for item in truth],
                [1.0] * len(truth),
                device=arguments.device,
            )

    encoder = DazzleCowEncoder(arguments.model, device=arguments.device)
    gallery = DazzleCowGallery(
        arguments.gallery,
        neighbors=arguments.neighbors,
        match_threshold=arguments.match_threshold,
        match_margin=arguments.match_margin,
    )
    observations = video_observations(
        arguments.video,
        arguments.annotations,
        localize,
        encoder,
        start_frame=arguments.start_frame,
        end_frame=arguments.end_frame,
        frame_step=arguments.frame_step,
        max_frames=arguments.max_frames,
    )
    report = {
        "settings": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(arguments).items()
            if key != "output"
        },
        "metrics": evaluate_observations(
            observations,
            gallery,
            sample_counts=arguments.track_samples,
            track_iou=arguments.track_iou,
            track_max_age=arguments.track_max_age,
            ground_truth_iou=arguments.ground_truth_iou,
        ),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["metrics"], indent=2, sort_keys=True))
