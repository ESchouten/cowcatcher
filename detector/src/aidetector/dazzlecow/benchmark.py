import argparse
import csv
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from aidetector.dazzlecow.gallery import DazzleCowGallery
from aidetector.dazzlecow.metrics import clustering_metrics
from aidetector.dazzlecow.model import DazzleCowEncoder
from aidetector.dazzlecow.train import (
    CowImage,
    build_gallery,
    discover_public_dataset,
    evaluate,
    train,
)

logger = logging.getLogger(__name__)


def encode_source(
    model: Path,
    source: Path,
    *,
    batch_size: int,
    device: str,
) -> list[tuple[CowImage, np.ndarray]]:
    samples = discover_public_dataset(f"identity={source}")
    encoder = DazzleCowEncoder(model, device=device)
    encoded = []
    for offset in range(0, len(samples), batch_size):
        batch = samples[offset : offset + batch_size]
        images = [cv2.imread(str(sample.path)) for sample in batch]
        valid = [
            (sample, image)
            for sample, image in zip(batch, images, strict=True)
            if image is not None
        ]
        embeddings = encoder.embed([image for _, image in valid])
        encoded.extend(
            (sample, embedding)
            for (sample, _), embedding in zip(valid, embeddings, strict=True)
        )
    return encoded


def track_aggregation_metrics(
    encoded: list[tuple[CowImage, np.ndarray]],
    gallery: DazzleCowGallery,
    sample_counts: list[int],
) -> dict[str, Any]:
    if any(count < 1 for count in sample_counts):
        raise ValueError("Track sample counts must be positive")
    by_identity: defaultdict[str, list[np.ndarray]] = defaultdict(list)
    for sample, embedding in encoded:
        by_identity[sample.identity.removeprefix("identity:")].append(embedding)

    metrics = {}
    for count in sorted(set(sample_counts)):
        queries = []
        for identity, embeddings in sorted(by_identity.items()):
            for end in range(count, len(embeddings) + 1):
                queries.append(
                    (
                        identity,
                        _normalize(np.mean(embeddings[end - count : end], axis=0)),
                    )
                )
        metrics[str(count)] = _query_metrics(queries, gallery)
    return metrics


def open_set_metrics(
    encoded: list[tuple[CowImage, np.ndarray]],
    gallery: DazzleCowGallery,
    configured_threshold: float,
    configured_margin: float,
) -> tuple[dict[str, Any], list[dict[str, float]]]:
    identities = sorted(
        {sample.identity.removeprefix("identity:") for sample, _ in encoded}
    )
    missing = set(identities) - set(gallery.identities)
    if missing:
        raise ValueError(f"Test identities missing from gallery: {sorted(missing)}")
    if len(set(gallery.identities)) < 2:
        raise ValueError("Open-set evaluation requires at least two gallery identities")

    raw_gallery = DazzleCowGallery.from_data(
        gallery.embeddings,
        gallery.identities,
        neighbors=gallery.neighbors,
        match_threshold=-1,
    )
    known = []
    unknown: list[tuple[str, float]] = []
    for identity in identities:
        identity_samples = [
            embedding
            for sample, embedding in encoded
            if sample.identity.removeprefix("identity:") == identity
        ]
        without_identity = gallery.identities != identity
        held_out_gallery = DazzleCowGallery.from_data(
            gallery.embeddings[without_identity],
            gallery.identities[without_identity],
            neighbors=gallery.neighbors,
            match_threshold=-1,
        )
        for embedding in identity_samples:
            known_score = raw_gallery.score(embedding)
            unknown_score = held_out_gallery.score(embedding)
            known.append(
                (
                    known_score.identity == identity,
                    known_score.similarity,
                    known_score.margin,
                )
            )
            unknown.append(
                (identity, unknown_score.similarity, unknown_score.margin)
            )

    sweep = [
        {
            "similarity_threshold": float(threshold),
            "margin_threshold": float(margin),
            **_threshold_metrics(
                known,
                unknown,
                float(threshold),
                float(margin),
            ),
        }
        for threshold in np.linspace(0, 1, 201)
        for margin in np.linspace(0, 1, 101)
    ]
    recommended = max(
        sweep,
        key=lambda row: (row["balanced_accuracy"], row["unknown_rejection_rate"]),
    )
    threshold = recommended["similarity_threshold"]
    margin = recommended["margin_threshold"]
    return (
        {
            "best_balanced_threshold": threshold,
            "best_balanced_margin": margin,
            "best_balanced": {
                key: value
                for key, value in recommended.items()
                if key not in {"similarity_threshold", "margin_threshold"}
            },
            "configured_threshold": configured_threshold,
            "configured_margin": configured_margin,
            "configured": _threshold_metrics(
                known,
                unknown,
                configured_threshold,
                configured_margin,
            ),
            "known_top1_accuracy": sum(correct for correct, _, _ in known)
            / len(known),
            "mean_unknown_similarity": float(
                np.mean([similarity for _, similarity, _ in unknown])
            ),
            "mean_unknown_margin": float(
                np.mean([margin for _, _, margin in unknown])
            ),
            "per_identity": {
                identity: {
                    "unknown_rejection_rate": sum(
                        similarity < threshold or score_margin < margin
                        for current, similarity, score_margin in unknown
                        if current == identity
                    )
                    / sum(current == identity for current, _, _ in unknown),
                    "mean_unknown_similarity": float(
                        np.mean(
                            [
                                similarity
                                for current, similarity, _ in unknown
                                if current == identity
                            ]
                        )
                    ),
                    "mean_unknown_margin": float(
                        np.mean(
                            [
                                score_margin
                                for current, _, score_margin in unknown
                                if current == identity
                            ]
                        )
                    ),
                }
                for identity in identities
            },
        },
        sweep,
    )


def _query_metrics(
    queries: list[tuple[str, np.ndarray]],
    gallery: DazzleCowGallery,
) -> dict[str, Any]:
    correct = 0
    matched = 0
    totals: Counter[str] = Counter()
    correct_by_identity: Counter[str] = Counter()
    similarities = []
    for expected, embedding in queries:
        totals[expected] += 1
        result = gallery.match(embedding)
        if result is None:
            continue
        matched += 1
        similarities.append(result.similarity)
        if result.identity == expected:
            correct += 1
            correct_by_identity[expected] += 1
    total = len(queries)
    return {
        "total": total,
        "matched": matched,
        "correct": correct,
        "accuracy": correct / total if total else 0,
        "coverage": matched / total if total else 0,
        "matched_accuracy": correct / matched if matched else 0,
        "mean_similarity": float(np.mean(similarities)) if similarities else 0,
        "per_identity": {
            identity: correct_by_identity[identity] / count
            for identity, count in sorted(totals.items())
        },
    }


def _threshold_metrics(
    known: list[tuple[bool, float, float]],
    unknown: list[tuple[str, float, float]],
    threshold: float,
    margin: float,
) -> dict[str, float]:
    known_rate = sum(
        correct and similarity >= threshold and score_margin >= margin
        for correct, similarity, score_margin in known
    ) / len(known)
    unknown_rate = sum(
        similarity < threshold or score_margin < margin
        for _, similarity, score_margin in unknown
    ) / len(unknown)
    return {
        "known_identification_rate": known_rate,
        "unknown_rejection_rate": unknown_rate,
        "unknown_false_acceptance_rate": 1 - unknown_rate,
        "balanced_accuracy": (known_rate + unknown_rate) / 2,
    }


def _normalize(embedding: np.ndarray) -> np.ndarray:
    embedding = np.asarray(embedding, dtype=np.float32)
    return embedding / max(float(np.linalg.norm(embedding)), np.finfo(np.float32).eps)


def parse_model(value: str) -> tuple[str, Path]:
    try:
        name, raw_path = value.split("=", 1)
    except ValueError as error:
        raise ValueError("Compared model must be NAME=PATH") from error
    if not name or not all(character.isalnum() or character in "-_" for character in name):
        raise ValueError("Compared model name may only contain letters, numbers, - and _")
    return name, Path(raw_path).expanduser().resolve()


def write_manifest(source: Path, output: Path) -> dict[str, Any]:
    source = source.expanduser().resolve()
    samples = discover_public_dataset(f"identity={source}")
    manifest = {
        "root": str(source),
        "count": len(samples),
        "identities": len({sample.identity for sample in samples}),
        "samples": [
            {
                "path": str(sample.path.relative_to(source)),
                "identity": sample.identity.removeprefix("identity:"),
            }
            for sample in samples
        ],
    }
    _write_json(output, manifest)
    return {key: value for key, value in manifest.items() if key != "samples"}


def run_benchmark(
    *,
    train_source: Path,
    validation_source: Path,
    gallery_source: Path,
    test_source: Path,
    output: Path,
    comparisons: dict[str, Path],
    epochs: int,
    identities_per_batch: int,
    images_per_identity: int,
    workers: int,
    learning_rate: float,
    temperature: float,
    patience: int,
    training_mode: str,
    batch_size: int,
    neighbors: int,
    match_threshold: float,
    match_margin: float,
    track_samples: list[int],
    device: str,
    seed: int,
) -> dict[str, Any]:
    output = output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Benchmark output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    sources = {
        "train": train_source,
        "validation": validation_source,
        "gallery": gallery_source,
        "test": test_source,
    }
    manifests = {
        name: write_manifest(source, output / "manifests" / f"{name}.json")
        for name, source in sources.items()
    }
    trained_model = output / "models" / "trained.pt"
    settings = {
        "seed": seed,
        "epochs": epochs,
        "identities_per_batch": identities_per_batch,
        "images_per_identity": images_per_identity,
        "workers": workers,
        "learning_rate": learning_rate,
        "temperature": temperature,
        "patience": patience,
        "training_mode": training_mode,
        "batch_size": batch_size,
        "neighbors": neighbors,
        "match_threshold": match_threshold,
        "match_margin": match_margin,
        "track_samples": track_samples,
        "device": device,
    }
    report: dict[str, Any] = {
        "settings": settings,
        "datasets": manifests,
        "models": {
            "trained": str(trained_model),
            **{name: str(path.resolve()) for name, path in comparisons.items()},
        },
        "metrics": {},
    }
    _write_json(output / "benchmark.json", report)

    train(
        [f"identity={train_source}"],
        trained_model,
        validation_specs=[f"identity={validation_source}"],
        epochs=epochs,
        identities_per_batch=identities_per_batch,
        images_per_identity=images_per_identity,
        workers=workers,
        learning_rate=learning_rate,
        temperature=temperature,
        patience=patience,
        training_mode=training_mode,
        device=device,
        seed=seed,
    )

    models = {"trained": trained_model, **comparisons}
    for name, model in models.items():
        logger.info("Evaluating benchmark model %s", name)
        gallery = output / "galleries" / f"{name}.npz"
        build_gallery(
            model,
            gallery_source,
            gallery,
            batch_size=batch_size,
            device=device,
        )
        frame_metrics = evaluate(
            model,
            gallery,
            test_source,
            gallery_source=gallery_source,
            failures_directory=output / "failures" / name,
            batch_size=batch_size,
            neighbors=neighbors,
            match_threshold=match_threshold,
            match_margin=match_margin,
            device=device,
        )
        encoded = encode_source(
            model,
            test_source,
            batch_size=batch_size,
            device=device,
        )
        identity_gallery = DazzleCowGallery(
            gallery,
            neighbors=neighbors,
            match_threshold=match_threshold,
            match_margin=match_margin,
        )
        open_set, threshold_sweep = open_set_metrics(
            encoded,
            identity_gallery,
            match_threshold,
            match_margin,
        )
        report["metrics"][name] = {
            "frame": frame_metrics,
            "clustering": clustering_metrics(
                np.asarray([embedding for _, embedding in encoded]),
                [
                    sample.identity.removeprefix("identity:")
                    for sample, _ in encoded
                ],
                neighbors=neighbors,
            ),
            "track_aggregation": track_aggregation_metrics(
                encoded,
                identity_gallery,
                track_samples,
            ),
            "open_set": open_set,
        }
        _write_csv(output / "open-set" / f"{name}.csv", threshold_sweep)
        _write_json(output / "benchmark.json", report)
    return report


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as file:
        json.dump(value, file, indent=2, sort_keys=True)
        file.write("\n")


def _write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and compare DazzleCow models on fixed dataset splits"
    )
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--gallery", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--compare",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Existing checkpoint to evaluate on the same gallery and test data",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--identities-per-batch", type=int, default=8)
    parser.add_argument("--images-per-identity", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument(
        "--training-mode",
        choices=("supervised", "paper"),
        default="supervised",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--neighbors", type=int, default=5)
    parser.add_argument("--match-threshold", type=float, default=0)
    parser.add_argument("--match-margin", type=float, default=0)
    parser.add_argument("--track-samples", type=int, nargs="+", default=[1, 3, 5, 8])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=84000)
    arguments = parser.parse_args()

    comparisons = dict(parse_model(value) for value in arguments.compare)
    if "trained" in comparisons:
        parser.error("The model name 'trained' is reserved")
    logging.basicConfig(level=logging.INFO)
    report = run_benchmark(
        train_source=arguments.train,
        validation_source=arguments.validation,
        gallery_source=arguments.gallery,
        test_source=arguments.test,
        output=arguments.output,
        comparisons=comparisons,
        epochs=arguments.epochs,
        identities_per_batch=arguments.identities_per_batch,
        images_per_identity=arguments.images_per_identity,
        workers=arguments.workers,
        learning_rate=arguments.learning_rate,
        temperature=arguments.temperature,
        patience=arguments.patience,
        training_mode=arguments.training_mode,
        batch_size=arguments.batch_size,
        neighbors=arguments.neighbors,
        match_threshold=arguments.match_threshold,
        match_margin=arguments.match_margin,
        track_samples=arguments.track_samples,
        device=arguments.device,
        seed=arguments.seed,
    )
    for name, metrics in report["metrics"].items():
        frame = metrics["frame"]
        print(
            f"{name}: {frame['accuracy']:.1%} frame top-1, "
            f"best balanced threshold "
            f"{metrics['open_set']['best_balanced_threshold']:.3f}, margin "
            f"{metrics['open_set']['best_balanced_margin']:.3f}"
        )
