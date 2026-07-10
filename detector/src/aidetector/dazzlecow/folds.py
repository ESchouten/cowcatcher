import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path

from aidetector.dazzlecow.train import CowImage, discover_public_dataset


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
