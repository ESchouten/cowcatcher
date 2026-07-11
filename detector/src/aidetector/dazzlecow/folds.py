import argparse
import json
import os
import random
import shutil
from collections import Counter
from pathlib import Path

from aidetector.dazzlecow.train import CowImage, discover_public_dataset


def create_identity_disjoint_fold(
    samples: list[CowImage],
    output: Path,
    *,
    validation_group: str,
    gallery_group: str,
    query_group: str,
    gallery_camera: str,
    query_cameras: set[str],
    train_identities: int,
    validation_known: int,
    validation_unknown: int,
    test_known: int,
    test_unknown: int,
    seed: int,
) -> dict:
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Fold output directory is not empty: {output}")

    by_identity: dict[str, list[CowImage]] = {}
    for sample in samples:
        identity = sample.identity.split(":", 1)[-1]
        by_identity.setdefault(identity, []).append(sample)

    validation_count = validation_known + validation_unknown
    test_count = test_known + test_unknown
    validation_eligible = [
        identity
        for identity, identity_samples in by_identity.items()
        if _has_view(identity_samples, validation_group)
        and _has_view(identity_samples, gallery_group, {gallery_camera})
        and _has_view(identity_samples, query_group, query_cameras)
    ]
    test_eligible = [
        identity
        for identity, identity_samples in by_identity.items()
        if _has_view(identity_samples, gallery_group, {gallery_camera})
        and _has_view(identity_samples, query_group, query_cameras)
    ]
    training_eligible = {
        identity
        for identity, identity_samples in by_identity.items()
        if any(
            sample.group not in {gallery_group, query_group}
            for sample in identity_samples
        )
    }
    generator = random.Random(seed)
    required_test = sorted(set(by_identity) - training_eligible)
    if not set(required_test) <= set(test_eligible):
        raise ValueError(
            "Identities without training images also lack required test views: "
            f"{sorted(set(required_test) - set(test_eligible))}"
        )
    if len(required_test) > test_count:
        raise ValueError(
            f"Test split has {test_count} places but requires "
            f"{len(required_test)} non-training identities"
        )

    test_candidates = sorted(set(test_eligible) - set(required_test))
    generator.shuffle(test_candidates)
    selected_test = required_test + test_candidates[: test_count - len(required_test)]
    generator.shuffle(selected_test)

    validation_candidates = sorted(set(validation_eligible) - set(selected_test))
    generator.shuffle(validation_candidates)
    if len(validation_candidates) < validation_count:
        raise ValueError(
            f"Need {validation_count} validation identities, "
            f"found {len(validation_candidates)} after selecting test identities"
        )
    selected_validation = validation_candidates[:validation_count]

    identities = {
        "validation_known": sorted(selected_validation[:validation_known]),
        "validation_unknown": sorted(selected_validation[validation_known:]),
        "test_known": sorted(selected_test[:test_known]),
        "test_unknown": sorted(selected_test[test_known:]),
    }
    held_out = {identity for values in identities.values() for identity in values}
    training_candidates = sorted(training_eligible - held_out)
    generator.shuffle(training_candidates)
    if len(training_candidates) < train_identities:
        raise ValueError(
            f"Need {train_identities} training identities, "
            f"found {len(training_candidates)}"
        )
    identities["train"] = sorted(training_candidates[:train_identities])

    split_identities = {
        "train": set(identities["train"]),
        "validation": set(
            identities["validation_known"] + identities["validation_unknown"]
        ),
        "calibration_gallery": set(identities["validation_known"]),
        "calibration_query": set(
            identities["validation_known"] + identities["validation_unknown"]
        ),
        "gallery": set(identities["test_known"]),
        "test": set(identities["test_known"] + identities["test_unknown"]),
    }
    counts = Counter()
    for identity, identity_samples in by_identity.items():
        for sample in identity_samples:
            camera = _camera(sample)
            targets = []
            if identity in split_identities["train"] and sample.group not in {
                gallery_group,
                query_group,
            }:
                targets.append("train")
            if (
                identity in split_identities["validation"]
                and sample.group == validation_group
            ):
                targets.append("validation")
            if (
                identity in split_identities["calibration_gallery"]
                and sample.group == gallery_group
                and camera == gallery_camera
            ):
                targets.append("calibration_gallery")
            if (
                identity in split_identities["calibration_query"]
                and sample.group == query_group
                and camera in query_cameras
            ):
                targets.append("calibration_query")
            if (
                identity in split_identities["gallery"]
                and sample.group == gallery_group
                and camera == gallery_camera
            ):
                targets.append("gallery")
            if (
                identity in split_identities["test"]
                and sample.group == query_group
                and camera in query_cameras
            ):
                targets.append("test")
            for split in targets:
                _link_sample(sample, output / split / identity)
                counts[split] += 1

    report = {
        "seed": seed,
        "validation_group": validation_group,
        "gallery_group": gallery_group,
        "query_group": query_group,
        "gallery_camera": gallery_camera,
        "query_cameras": sorted(query_cameras),
        "identities": identities,
        "counts": dict(counts),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "fold.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def _has_view(
    samples: list[CowImage],
    group: str,
    cameras: set[str] | None = None,
) -> bool:
    return any(
        sample.group == group and (cameras is None or _camera(sample) in cameras)
        for sample in samples
    )


def _camera(sample: CowImage) -> str:
    return sample.path.stem.rsplit("_", 1)[-1]


def _link_sample(sample: CowImage, directory: Path) -> None:
    group = (sample.group or "group").replace("/", "_")
    target = directory / f"{group}--{sample.path.name}"
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(sample.path, target)
    except OSError:
        shutil.copy2(sample.path, target)


def create_fold(
    samples: list[CowImage],
    output: Path,
    *,
    validation_group: str,
    test_group: str,
) -> dict:
    if validation_group == test_group:
        raise ValueError("Validation and test groups must differ")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Fold output directory is not empty: {output}")

    counts = Counter()
    for sample in samples:
        if sample.group is None:
            raise ValueError(f"Sample has no day/camera group: {sample.path}")
        split = (
            "test"
            if sample.group == test_group
            else "validation"
            if sample.group == validation_group
            else "train"
        )
        identity = sample.identity.split(":", 1)[-1]
        group = sample.group.replace("/", "_")
        timestamp = (sample.timestamp or sample.path.stem).replace("/", "_")
        filename = f"{group}--{timestamp}--{sample.path.name}"
        targets = [output / split / identity / filename]
        if split == "validation":
            targets.append(output / "gallery" / identity / filename)
        for target in targets:
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(sample.path, target)
            except OSError:
                shutil.copy2(sample.path, target)
        counts[split] += 1

    report = {
        "validation_group": validation_group,
        "test_group": test_group,
        "counts": dict(counts),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "fold.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create day/camera-isolated DazzleCow benchmark folds"
    )
    parser.add_argument("--dataset", required=True, help="TYPE=PATH")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold", type=int, help="One-based test-group fold to create")
    arguments = parser.parse_args()

    samples = discover_public_dataset(arguments.dataset)
    groups = sorted({sample.group for sample in samples if sample.group is not None})
    if len(groups) < 3:
        raise ValueError("At least three day/camera groups are required")
    selected = range(len(groups)) if arguments.fold is None else [arguments.fold - 1]
    for index in selected:
        if not 0 <= index < len(groups):
            raise ValueError(f"Fold must be between 1 and {len(groups)}")
        create_fold(
            samples,
            arguments.output / f"fold-{index + 1:02d}",
            validation_group=groups[index - 1],
            test_group=groups[index],
        )


def identity_disjoint_main() -> None:
    parser = argparse.ArgumentParser(
        description="Create identity, day, and camera-disjoint DazzleCow folds"
    )
    parser.add_argument("--dataset", required=True, help="TYPE=PATH")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-group", default="2023Aug18")
    parser.add_argument("--gallery-group", default="2023Aug19")
    parser.add_argument("--query-group", default="2023Aug20")
    parser.add_argument("--gallery-camera", default="1")
    parser.add_argument("--query-cameras", nargs="+", default=["2", "3"])
    parser.add_argument("--train-identities", type=int, default=50)
    parser.add_argument("--validation-known", type=int, default=10)
    parser.add_argument("--validation-unknown", type=int, default=5)
    parser.add_argument("--test-known", type=int, default=20)
    parser.add_argument("--test-unknown", type=int, default=5)
    parser.add_argument("--seed", type=int, default=84000)
    arguments = parser.parse_args()
    report = create_identity_disjoint_fold(
        discover_public_dataset(arguments.dataset),
        arguments.output,
        validation_group=arguments.validation_group,
        gallery_group=arguments.gallery_group,
        query_group=arguments.query_group,
        gallery_camera=arguments.gallery_camera,
        query_cameras=set(arguments.query_cameras),
        train_identities=arguments.train_identities,
        validation_known=arguments.validation_known,
        validation_unknown=arguments.validation_unknown,
        test_known=arguments.test_known,
        test_unknown=arguments.test_unknown,
        seed=arguments.seed,
    )
    print(json.dumps(report["counts"], indent=2))
