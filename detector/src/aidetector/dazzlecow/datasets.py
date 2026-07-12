from dataclasses import dataclass
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class CowImage:
    path: Path
    identity: str
    timestamp: str | None = None
    group: str | None = None


def discover_public_dataset(specification: str) -> list[CowImage]:
    try:
        dataset_type, raw_root = specification.split("=", 1)
    except ValueError as error:
        raise ValueError(
            "Dataset must be TYPE=PATH, for example multicamcows2024=/data/cows"
        ) from error

    root = Path(raw_root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Dataset directory does not exist: {root}")

    samples = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        relative = path.relative_to(root)
        identity = _identity_from_path(dataset_type, relative)
        if identity is not None:
            samples.append(
                CowImage(
                    path,
                    f"{dataset_type}:{identity}",
                    _timestamp_from_path(dataset_type, relative),
                    _group_from_path(dataset_type, relative),
                )
            )
    return samples


def _identity_from_path(dataset_type: str, relative: Path) -> str | None:
    parts = relative.parts
    if dataset_type == "multicamcows2024":
        return parts[-2] if len(parts) >= 3 else None
    if dataset_type == "cows2021":
        return next((part for part in parts[:-1] if part.isdigit()), None)
    if dataset_type == "identity":
        return parts[0] if len(parts) >= 2 else None
    raise ValueError(f"Unknown public cow dataset type: {dataset_type}")


def _timestamp_from_path(dataset_type: str, relative: Path) -> str:
    stem = relative.stem
    if dataset_type == "multicamcows2024":
        return "/".join((*relative.parts[:-2], stem))
    if dataset_type == "identity" and stem.count("--") >= 2:
        group, timestamp, _ = stem.split("--", 2)
        return f"{group}/{timestamp}"
    if len(stem) > 9 and stem[:8].isdigit() and stem[8] == "-":
        return stem[9:]
    return stem


def _group_from_path(dataset_type: str, relative: Path) -> str | None:
    if dataset_type == "multicamcows2024" and len(relative.parts) >= 3:
        return relative.parts[0]
    if dataset_type == "identity" and relative.stem.count("--") >= 2:
        return relative.stem.split("--", 1)[0]
    return None
