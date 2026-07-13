import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from aidetector.dazzlecow.enrollment import (
    DEFAULT_ENROLLMENT_MARGIN,
    DEFAULT_ENROLLMENT_SIMILARITY,
    EnrollmentTrack,
    cluster_known_count,
    cluster_tracklets,
    cluster_tracks,
    match_camera_tracks,
)
from aidetector.dazzlecow.gallery import CowIdentityGallery
from aidetector.dazzlecow.datasets import discover_public_dataset
from aidetector.dazzlecow.model import CowIdentityEncoder, IDENTITY_MODEL
from aidetector.domain.vectors import normalize_vector
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score


@dataclass(frozen=True)
class LabeledTrack:
    track: EnrollmentTrack
    identity: str
    camera: str
    frames: frozenset[int] = frozenset()


def build_track_embeddings(
    source: Path,
    *,
    samples_per_track: int,
    fragments_per_sequence: int,
    batch_size: int,
    synchronized_frames: bool = True,
) -> list[LabeledTrack]:
    if fragments_per_sequence < 1:
        raise ValueError("fragments_per_sequence must be at least 1")

    grouped = defaultdict(list)
    for sample in discover_public_dataset(f"identity={source}"):
        identity = sample.identity.removeprefix("identity:")
        camera = _camera_from_path(sample.path)
        grouped[(identity, camera)].append(sample.path)

    fragments = {}
    for (identity, camera), paths in sorted(grouped.items()):
        for fragment, chunk in enumerate(
            np.array_split(sorted(paths), fragments_per_sequence)
        ):
            if len(chunk):
                fragments[(identity, camera, fragment)] = list(chunk)

    selected = []
    for group, paths in fragments.items():
        count = min(samples_per_track, len(paths))
        indices = np.linspace(0, len(paths) - 1, count, dtype=int)
        selected.extend((group, paths[index]) for index in indices)

    encoder = CowIdentityEncoder()
    embeddings = defaultdict(list)
    for offset in range(0, len(selected), batch_size):
        batch = selected[offset : offset + batch_size]
        images = [cv2.imread(str(path)) for _, path in batch]
        valid = [
            (group, image)
            for (group, _), image in zip(batch, images, strict=True)
            if image is not None
        ]
        encoded = encoder.embed([image for _, image in valid])
        for (group, _), embedding in zip(valid, encoded, strict=True):
            embeddings[group].append(embedding)

    keys = {
        group: f"track-{index:04d}" for index, group in enumerate(sorted(embeddings), 1)
    }
    frames = {
        group: frozenset(_frame_from_path(path) for path in paths)
        for group, paths in fragments.items()
    }
    intervals = {
        group: (min(group_frames), max(group_frames))
        for group, group_frames in frames.items()
    }

    tracks = []
    for group, values in sorted(embeddings.items()):
        identity, camera, _ = group
        embedding = normalize_vector(np.mean(values, axis=0))
        cannot_link = frozenset(
            keys[other]
            for other in embeddings
            if other != group
            and other[1] == camera
            and synchronized_frames
            and max(intervals[group][0], intervals[other][0])
            <= min(intervals[group][1], intervals[other][1])
        )
        tracks.append(
            LabeledTrack(
                EnrollmentTrack(keys[group], embedding, cannot_link),
                identity,
                camera,
                frames[group],
            )
        )
    return tracks


def enrollment_metrics(
    tracks: list[LabeledTrack],
    assignments: dict[str, str],
) -> dict[str, float | int]:
    true_by_key = {track.track.key: track.identity for track in tracks}
    keys = sorted(true_by_key)
    true_positive = false_positive = false_negative = 0
    for left_index, left in enumerate(keys):
        for right in keys[left_index + 1 :]:
            same_true = true_by_key[left] == true_by_key[right]
            same_predicted = assignments[left] == assignments[right]
            true_positive += same_true and same_predicted
            false_positive += not same_true and same_predicted
            false_negative += same_true and not same_predicted

    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 1.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 1.0
    )
    pair_f1 = (
        2 * precision * recall / (precision + recall) if precision + recall else 0.0
    )
    true_by_cluster = defaultdict(Counter)
    clusters_by_true = defaultdict(set)
    for key, cluster in assignments.items():
        identity = true_by_key[key]
        true_by_cluster[cluster][identity] += 1
        clusters_by_true[identity].add(cluster)
    purity = sum(max(counts.values()) for counts in true_by_cluster.values()) / len(
        tracks
    )
    complete_identities = sum(
        len(clusters) == 1 and len(true_by_cluster[next(iter(clusters))]) == 1
        for clusters in clusters_by_true.values()
    )
    true_identities = len(set(true_by_key.values()))
    return {
        "tracks": len(tracks),
        "true_identities": true_identities,
        "clusters": len(set(assignments.values())),
        "identities_per_true_identity": sum(
            len(clusters) for clusters in clusters_by_true.values()
        )
        / true_identities,
        "complete_identities": complete_identities,
        "complete_identity_rate": complete_identities / true_identities,
        "pair_precision": precision,
        "pair_recall": recall,
        "pair_f1": pair_f1,
        "purity": purity,
        "merge_errors": sum(len(counts) > 1 for counts in true_by_cluster.values()),
        "fragmented_identities": sum(
            len(clusters) > 1 for clusters in clusters_by_true.values()
        ),
    }


def production_open_enrollment_metrics(
    tracks: list[LabeledTrack],
) -> dict[str, float | int]:
    assignments = cluster_tracklets(
        [track.track for track in tracks],
        similarity_threshold=DEFAULT_ENROLLMENT_SIMILARITY,
        margin_threshold=DEFAULT_ENROLLMENT_MARGIN,
    )
    return enrollment_metrics(tracks, assignments)


def verification_metrics(tracks: list[LabeledTrack]) -> dict[str, float | int]:
    embeddings = np.asarray([track.track.embedding for track in tracks])
    similarities = embeddings @ embeddings.T
    positives = []
    negatives = []
    for left in range(len(tracks)):
        for right in range(left + 1, len(tracks)):
            target = tracks[left].identity == tracks[right].identity
            (positives if target else negatives).append(similarities[left, right])
    if not positives or not negatives:
        return {"positive_pairs": len(positives), "negative_pairs": len(negatives)}

    positive_scores = np.asarray(positives)
    negative_scores = np.asarray(negatives)
    safe_threshold = float(np.nextafter(negative_scores.max(), np.inf))
    return {
        "positive_pairs": len(positives),
        "negative_pairs": len(negatives),
        "roc_auc": float(
            roc_auc_score(
                np.concatenate(
                    (np.ones(len(positive_scores)), np.zeros(len(negative_scores)))
                ),
                np.concatenate((positive_scores, negative_scores)),
            )
        ),
        "safe_threshold": safe_threshold,
        "safe_positive_recall": float(np.mean(positive_scores >= safe_threshold)),
        "mean_positive_similarity": float(positive_scores.mean()),
        "mean_negative_similarity": float(negative_scores.mean()),
    }


def known_count_metrics(tracks: list[LabeledTrack]) -> dict[str, float | int]:
    identities = {track.identity for track in tracks}
    embeddings = np.asarray([track.track.embedding for track in tracks])
    labels = KMeans(
        n_clusters=len(identities),
        n_init=50,
        max_iter=500,
        random_state=84000,
    ).fit_predict(embeddings)
    assignments = {
        track.track.key: str(label) for track, label in zip(tracks, labels, strict=True)
    }
    return enrollment_metrics(tracks, assignments)


def camera_disjoint_identity_metrics(
    tracks: list[LabeledTrack],
    *,
    unknown_tracks: list[LabeledTrack] | None = None,
    match_threshold: float = 0.7,
    match_margin: float = 0.2,
) -> dict:
    totals = Counter()
    per_camera = {}
    for camera in sorted({track.camera for track in tracks}):
        gallery_tracks = [track for track in tracks if track.camera != camera]
        gallery_identities = {track.identity for track in gallery_tracks}
        queries = [
            track
            for track in tracks
            if track.camera == camera and track.identity in gallery_identities
        ]
        if not gallery_tracks or not queries:
            continue

        gallery = CowIdentityGallery(
            np.asarray([track.track.embedding for track in gallery_tracks]),
            [track.identity for track in gallery_tracks],
            [track.identity for track in gallery_tracks],
            match_threshold=match_threshold,
            match_margin=match_margin,
        )
        counts = Counter()
        for query in queries:
            score = gallery.score(query.track.embedding)
            accepted = (
                score.similarity >= match_threshold and score.margin >= match_margin
            )
            counts["queries"] += 1
            counts["top1"] += score.key == query.identity
            counts["identified"] += accepted and score.key == query.identity
            counts["misidentified"] += accepted and score.key != query.identity
            counts["rejected"] += not accepted

        for query in unknown_tracks or []:
            if query.camera != camera or query.identity in gallery_identities:
                continue
            score = gallery.score(query.track.embedding)
            counts["unknown_queries"] += 1
            counts["unknown_false_accepts"] += (
                score.similarity >= match_threshold and score.margin >= match_margin
            )

        totals.update(counts)
        per_camera[camera] = _identity_rates(counts)

    return {
        "match_threshold": match_threshold,
        "match_margin": match_margin,
        **_identity_rates(totals),
        "per_camera": per_camera,
    }


def _identity_rates(counts: Counter) -> dict:
    queries = counts["queries"]
    unknown = counts["unknown_queries"]
    return {
        "queries": queries,
        "top1_accuracy": counts["top1"] / queries if queries else None,
        "identification_rate": counts["identified"] / queries if queries else None,
        "misidentification_rate": (
            counts["misidentified"] / queries if queries else None
        ),
        "rejection_rate": counts["rejected"] / queries if queries else None,
        "unknown_queries": unknown,
        "unknown_false_acceptance_rate": (
            counts["unknown_false_accepts"] / unknown if unknown else None
        ),
    }


def constrained_known_count_metrics(
    tracks: list[LabeledTrack],
    *,
    attempts: int,
) -> dict[str, float | int]:
    assignments = cluster_known_count(
        [track.track for track in tracks],
        len({track.identity for track in tracks}),
        attempts=attempts,
    )
    return enrollment_metrics(tracks, assignments)


def evaluate_threshold(
    tracks: list[LabeledTrack],
    threshold: float,
    margin: float,
    neighbors: int,
    strategy: str,
) -> tuple[dict[str, str], dict[str, float | int]]:
    if strategy == "hungarian":
        assignments = match_camera_tracks(
            [track.track for track in tracks],
            {track.track.key: track.camera for track in tracks},
            similarity_threshold=threshold,
            margin_threshold=margin,
        )
    elif strategy == "mutual":
        assignments = cluster_tracks(
            [track.track for track in tracks],
            similarity_threshold=threshold,
            neighbors=neighbors,
        )
    else:
        assignments = cluster_tracklets(
            [track.track for track in tracks],
            similarity_threshold=threshold,
            margin_threshold=margin,
        )
    return assignments, enrollment_metrics(tracks, assignments)


def run_enrollment_benchmark(
    *,
    calibration_source: Path | None,
    test_source: Path,
    output: Path,
    samples_per_track: int,
    fragments_per_sequence: int,
    batch_size: int,
    neighbors: int,
    strategy: str,
    similarity_threshold: float | None = None,
    margin_threshold: float = 0.0,
    clustering_attempts: int = 1000,
    synchronized_frames: bool = True,
    identity_match_threshold: float = 0.68,
    identity_match_margin: float = 0.05,
) -> dict:
    test_tracks = build_track_embeddings(
        test_source,
        samples_per_track=samples_per_track,
        fragments_per_sequence=fragments_per_sequence,
        batch_size=batch_size,
        synchronized_frames=synchronized_frames,
    )
    calibration_tracks = (
        build_track_embeddings(
            calibration_source,
            samples_per_track=samples_per_track,
            fragments_per_sequence=fragments_per_sequence,
            batch_size=batch_size,
            synchronized_frames=synchronized_frames,
        )
        if calibration_source is not None
        else None
    )
    calibration_best = None
    calibration_selected = None
    if similarity_threshold is None:
        if calibration_tracks is None:
            raise ValueError("Calibration source or similarity threshold is required")
        calibration = []
        thresholds, margins = _calibration_grid(strategy)
        for threshold in thresholds:
            for margin in margins:
                _, metrics = evaluate_threshold(
                    calibration_tracks,
                    float(threshold),
                    float(margin),
                    neighbors,
                    strategy,
                )
                calibration.append(
                    {
                        "threshold": float(threshold),
                        "margin": float(margin),
                        **metrics,
                    }
                )
        calibration_best = select_enrollment_threshold(calibration)
        calibration_selected = select_enrollment_threshold(
            calibration,
            margin_buffer=(0.01 if strategy in {"agglomerative", "hungarian"} else 0.0),
        )
        similarity_threshold = calibration_selected["threshold"]
        margin_threshold = calibration_selected["margin"]
    assignments, test = evaluate_threshold(
        test_tracks,
        similarity_threshold,
        margin_threshold,
        neighbors,
        strategy,
    )
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "model": IDENTITY_MODEL,
        "samples_per_track": samples_per_track,
        "fragments_per_sequence": fragments_per_sequence,
        "synchronized_frames": synchronized_frames,
        "neighbors": neighbors,
        "strategy": strategy,
        "selected_threshold": similarity_threshold,
        "selected_margin": margin_threshold,
        "calibration_best": calibration_best,
        "calibration": calibration_selected,
        "test": test,
        "production_open_enrollment": production_open_enrollment_metrics(test_tracks),
        "verification": verification_metrics(test_tracks),
        "camera_disjoint_identity": camera_disjoint_identity_metrics(
            test_tracks,
            unknown_tracks=calibration_tracks,
            match_threshold=identity_match_threshold,
            match_margin=identity_match_margin,
        ),
        "known_identity_count": known_count_metrics(test_tracks),
        "constrained_known_identity_count": constrained_known_count_metrics(
            test_tracks,
            attempts=clustering_attempts,
        ),
    }
    (output / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def select_enrollment_threshold(
    results: list[dict], *, margin_buffer: float = 0.0
) -> dict:
    safe = [result for result in results if result["merge_errors"] == 0]
    candidates = safe or results
    best = max(
        candidates,
        key=lambda result: (
            -result["merge_errors"],
            result["pair_recall"],
            result["pair_precision"],
            -result["clusters"],
        ),
    )
    guarded = [
        result
        for result in candidates
        if result["threshold"] >= best["threshold"]
        and result["margin"] >= best["margin"] + margin_buffer
    ]
    if not guarded:
        return best
    return max(
        guarded,
        key=lambda result: (
            result["pair_recall"],
            result["pair_precision"],
            -result["clusters"],
            -result["threshold"],
            -result["margin"],
        ),
    )


def _camera_from_path(path: Path) -> str:
    _, separator, suffix = path.stem.rpartition("_")
    return suffix if separator and suffix.isdigit() else "1"


def _frame_from_path(path: Path) -> int:
    prefix = path.stem.split("_", 1)[0]
    return int(prefix) if prefix.isdigit() else 0


def _calibration_grid(strategy: str) -> tuple[np.ndarray, np.ndarray]:
    if strategy == "agglomerative":
        thresholds = np.unique(
            np.concatenate((np.linspace(0.7, 0.99, 30), [0.995, 0.999]))
        )
        margins = np.asarray([0, 0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01])
        return thresholds, margins
    if strategy == "hungarian":
        return np.linspace(0.7, 0.999, 300), np.linspace(0, 0.2, 21)
    return np.linspace(0.7, 0.999, 300), np.asarray([0.0])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate and evaluate automatic cow enrollment"
    )
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-track", type=int, default=8)
    parser.add_argument("--fragments-per-sequence", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--neighbors", type=int, default=5)
    parser.add_argument(
        "--strategy",
        choices=("agglomerative", "hungarian", "mutual"),
        default="agglomerative",
    )
    parser.add_argument("--similarity-threshold", type=float)
    parser.add_argument("--margin-threshold", type=float, default=0.0)
    parser.add_argument("--identity-match-threshold", type=float, default=0.68)
    parser.add_argument("--identity-match-margin", type=float, default=0.05)
    parser.add_argument("--clustering-attempts", type=int, default=1000)
    parser.add_argument("--independent-sequences", action="store_true")
    arguments = parser.parse_args()
    report = run_enrollment_benchmark(
        calibration_source=arguments.calibration,
        test_source=arguments.test,
        output=arguments.output,
        samples_per_track=arguments.samples_per_track,
        fragments_per_sequence=arguments.fragments_per_sequence,
        batch_size=arguments.batch_size,
        neighbors=arguments.neighbors,
        strategy=arguments.strategy,
        similarity_threshold=arguments.similarity_threshold,
        margin_threshold=arguments.margin_threshold,
        clustering_attempts=arguments.clustering_attempts,
        synchronized_frames=not arguments.independent_sequences,
        identity_match_threshold=arguments.identity_match_threshold,
        identity_match_margin=arguments.identity_match_margin,
    )
    print(json.dumps(report, indent=2))
