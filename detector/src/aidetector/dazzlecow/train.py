import argparse
import csv
import logging
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np
import torch
from aidetector.dazzlecow.gallery import DazzleCowGallery, save_gallery
from aidetector.dazzlecow.model import (
    DazzleCowEncoder,
    IMAGENET_MEAN,
    IMAGENET_STD,
    create_encoder_model,
    resolve_device,
)
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms

logger = logging.getLogger(__name__)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
AUGMENTED_IMAGE_SIZE = 128


@dataclass(frozen=True)
class CowImage:
    path: Path
    identity: str
    timestamp: str | None = None
    group: str | None = None


class PublicCowDataset(Dataset):
    def __init__(
        self,
        samples: list[CowImage],
        *,
        image_size: int = 256,
        augment: bool = True,
    ):
        if not samples:
            raise ValueError("No cow images found")
        self.samples = samples
        self.augment = augment
        identities = sorted({sample.identity for sample in samples})
        self.label_by_identity = {
            identity: index for index, identity in enumerate(identities)
        }
        self.base_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN[:, 0, 0], IMAGENET_STD[:, 0, 0]),
            ]
        )
        self.augment_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    min(image_size, AUGMENTED_IMAGE_SIZE),
                    scale=(0.95, 1.0),
                    ratio=(0.95, 1.05),
                ),
                transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
                transforms.ElasticTransform(alpha=100.0),
                transforms.RandomAffine(
                    degrees=10,
                    translate=(0.1, 0.1),
                    scale=(0.9, 1.1),
                ),
                transforms.ColorJitter(
                    brightness=0.5,
                    contrast=0.5,
                    saturation=0.5,
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN[:, 0, 0], IMAGENET_STD[:, 0, 0]),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = Image.open(sample.path).convert("RGB")
        base = self.base_transform(image)
        label = self.label_by_identity[sample.identity]
        if not self.augment:
            return base, label
        return base, self.augment_transform(image), label


class IdentityBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        samples: list[CowImage],
        *,
        identities_per_batch: int,
        images_per_identity: int,
        seed: int,
    ):
        if identities_per_batch < 1 or images_per_identity < 1:
            raise ValueError("P x K batch dimensions must be positive")
        self.indices_by_identity: dict[str, list[int]] = defaultdict(list)
        for index, sample in enumerate(samples):
            self.indices_by_identity[sample.identity].append(index)
        self.identities = sorted(self.indices_by_identity)
        self.identities_per_batch = min(identities_per_batch, len(self.identities))
        self.images_per_identity = images_per_identity
        self.seed = seed
        self.epoch = 0

    def __iter__(self) -> Iterator[list[int]]:
        generator = random.Random(self.seed + self.epoch)
        self.epoch += 1
        for _ in range(len(self)):
            identities = generator.sample(
                self.identities,
                self.identities_per_batch,
            )
            batch = []
            for identity in identities:
                indices = self.indices_by_identity[identity]
                select = (
                    generator.sample
                    if len(indices) >= self.images_per_identity
                    else generator.choices
                )
                batch.extend(select(indices, k=self.images_per_identity))
            generator.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        batch_size = self.identities_per_batch * self.images_per_identity
        return math.ceil(sum(map(len, self.indices_by_identity.values())) / batch_size)


class TimestampBatchSampler(Sampler[list[int]]):
    def __init__(self, samples: list[CowImage], *, seed: int):
        groups: defaultdict[str, dict[str, int]] = defaultdict(dict)
        for index, sample in enumerate(samples):
            timestamp = sample.timestamp or sample.path.stem
            groups[timestamp].setdefault(sample.identity, index)
        self.groups = [
            list(group.values()) for group in groups.values() if len(group) > 1
        ]
        if not self.groups:
            raise ValueError(
                "Paper training requires timestamps containing multiple cows"
            )
        self.seed = seed
        self.epoch = 0

    def __iter__(self) -> Iterator[list[int]]:
        generator = random.Random(self.seed + self.epoch)
        self.epoch += 1
        groups = [list(group) for group in self.groups]
        generator.shuffle(groups)
        for group in groups:
            generator.shuffle(group)
            yield group

    def __len__(self) -> int:
        return len(self.groups)


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

    images = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    samples = []
    for path in images:
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


def supervised_nt_xent(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.5,
) -> torch.Tensor:
    embeddings = torch.nn.functional.normalize(embeddings, dim=1)
    logits = embeddings @ embeddings.T / temperature
    self_mask = torch.eye(len(logits), dtype=torch.bool, device=logits.device)
    positive_mask = labels[:, None].eq(labels[None, :]) & ~self_mask
    logits = logits.masked_fill(self_mask, torch.finfo(logits.dtype).min)
    log_probabilities = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    positive_count = positive_mask.sum(dim=1).clamp_min(1)
    return -((log_probabilities * positive_mask).sum(dim=1) / positive_count).mean()


def train(
    dataset_specs: list[str],
    output: Path,
    *,
    validation_specs: list[str] | None,
    epochs: int,
    identities_per_batch: int,
    images_per_identity: int,
    workers: int,
    learning_rate: float,
    temperature: float,
    patience: int,
    training_mode: str,
    device: str,
    seed: int,
    architecture: str = "resnet50",
    image_size: int = 256,
    resume: Path | None = None,
) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    samples = [
        sample
        for specification in dataset_specs
        for sample in discover_public_dataset(specification)
    ]
    dataset = PublicCowDataset(samples, image_size=image_size)
    sampler = (
        TimestampBatchSampler(samples, seed=seed)
        if training_mode == "paper"
        else IdentityBatchSampler(
            samples,
            identities_per_batch=identities_per_batch,
            images_per_identity=images_per_identity,
            seed=seed,
        )
    )
    target_device = resolve_device(device)
    pin_memory = target_device.type == "cuda"
    mixed_precision = target_device.type in {"cuda", "mps"}
    scaler = torch.amp.GradScaler(target_device.type, enabled=mixed_precision)
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=workers,
        persistent_workers=workers > 0,
        pin_memory=pin_memory,
    )
    validation_dataset = (
        PublicCowDataset(
            [
                sample
                for specification in validation_specs
                for sample in discover_public_dataset(specification)
            ],
            image_size=image_size,
            augment=False,
        )
        if validation_specs
        else None
    )
    model = create_encoder_model(
        feature_dim=128,
        pretrained=True,
        architecture=architecture,
    ).to(target_device)
    optimizer = (
        torch.optim.SGD(model.parameters(), lr=learning_rate)
        if training_mode == "paper"
        else torch.optim.AdamW(model.parameters(), lr=learning_rate)
    )
    if training_mode == "paper":
        try:
            from pytorch_metric_learning import losses, miners
        except ImportError as error:
            raise RuntimeError(
                "Paper training requires the 'dazzlecow' optional dependencies"
            ) from error
        paper_loss = losses.NTXentLoss(temperature=temperature)
        paper_miner = miners.MultiSimilarityMiner()
    output.parent.mkdir(parents=True, exist_ok=True)

    best_accuracy = -1.0
    stale_epochs = 0
    start_epoch = 0
    if resume is not None:
        payload = torch.load(resume, map_location="cpu", weights_only=False)
        if payload.get("architecture") != architecture:
            raise ValueError("Resume checkpoint architecture does not match")
        if int(payload.get("image_size", 256)) != image_size:
            raise ValueError("Resume checkpoint image size does not match")
        model.load_state_dict(payload["state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        scaler.load_state_dict(payload["scaler_state_dict"])
        start_epoch = int(payload["epoch"])
        best_accuracy = float(payload.get("best_accuracy", -1.0))
        stale_epochs = int(payload.get("stale_epochs", 0))
        sampler.epoch = start_epoch
        logger.info("Resuming DazzleCow after epoch %d", start_epoch)

    logger.info(
        "Training DazzleCow on %d images from %d identities with %s sampling (%d batches)",
        len(dataset),
        len(dataset.label_by_identity),
        training_mode,
        len(sampler),
    )
    completed_epoch = start_epoch
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_losses = []
        for batch_index, (base, augmented, labels) in enumerate(loader, 1):
            base = base.to(target_device, non_blocking=pin_memory)
            augmented = augmented.to(target_device, non_blocking=pin_memory)
            with torch.autocast(
                target_device.type,
                dtype=torch.float16,
                enabled=mixed_precision,
            ):
                embeddings = torch.cat((model(base), model(augmented)))
                if training_mode == "paper":
                    labels = torch.arange(len(base), device=target_device)
                    pair_labels = torch.cat((labels, labels))
                    hard_pairs = paper_miner(embeddings, pair_labels)
                    loss = paper_loss(embeddings, pair_labels, hard_pairs)
                else:
                    labels = labels.to(target_device)
                    pair_labels = torch.cat((labels, labels))
                    loss = supervised_nt_xent(embeddings, pair_labels, temperature)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_losses.append(loss.detach())
            if batch_index % 100 == 0:
                logger.info(
                    "Epoch %d/%d batch %d/%d",
                    epoch + 1,
                    epochs,
                    batch_index,
                    len(loader),
                )

        loss = float(torch.stack(epoch_losses).mean().cpu())
        completed_epoch = epoch + 1
        if validation_dataset is None:
            logger.info("Epoch %d/%d loss %.4f", epoch + 1, epochs, loss)
        else:
            accuracy = validation_accuracy(
                model,
                validation_dataset,
                target_device,
                clustering=training_mode == "paper",
                workers=workers,
            )
            logger.info(
                "Epoch %d/%d loss %.4f validation %s %.1f%%",
                epoch + 1,
                epochs,
                loss,
                "Hungarian" if training_mode == "paper" else "top-1",
                accuracy * 100,
            )
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                stale_epochs = 0
                save_checkpoint(
                    model,
                    output,
                    dataset_specs,
                    architecture=architecture,
                    image_size=image_size,
                    epoch=epoch + 1,
                    validation_accuracy=accuracy,
                )
            else:
                stale_epochs += 1

        save_training_checkpoint(
            model,
            optimizer,
            scaler,
            output.with_name(f"{output.stem}.last{output.suffix}"),
            architecture=architecture,
            image_size=image_size,
            epoch=completed_epoch,
            best_accuracy=best_accuracy,
            stale_epochs=stale_epochs,
        )
        if validation_dataset is not None and patience and stale_epochs >= patience:
            logger.info("Stopping after %d epochs without improvement", patience)
            break

    if validation_dataset is not None and best_accuracy >= 0:
        return
    save_checkpoint(
        model,
        output,
        dataset_specs,
        architecture=architecture,
        image_size=image_size,
        epoch=completed_epoch,
    )


def save_checkpoint(
    model,
    output: Path,
    dataset_specs: list[str],
    *,
    architecture: str,
    image_size: int,
    epoch: int,
    validation_accuracy: float | None = None,
) -> None:
    torch.save(
        {
            "architecture": architecture,
            "feature_dim": 128,
            "image_size": image_size,
            "state_dict": {
                name: value.detach().cpu() for name, value in model.state_dict().items()
            },
            "datasets": dataset_specs,
            "epoch": epoch,
            "validation_accuracy": validation_accuracy,
        },
        output,
    )


def save_training_checkpoint(
    model,
    optimizer,
    scaler,
    output: Path,
    *,
    architecture: str,
    image_size: int,
    epoch: int,
    best_accuracy: float,
    stale_epochs: int,
) -> None:
    torch.save(
        {
            "architecture": architecture,
            "image_size": image_size,
            "state_dict": {
                name: value.detach().cpu() for name, value in model.state_dict().items()
            },
            "optimizer_state_dict": _to_cpu(optimizer.state_dict()),
            "scaler_state_dict": scaler.state_dict(),
            "epoch": epoch,
            "best_accuracy": best_accuracy,
            "stale_epochs": stale_epochs,
        },
        output,
    )


def _to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    return value


def validation_accuracy(
    model,
    dataset: PublicCowDataset,
    device: torch.device,
    *,
    neighbors: int = 5,
    batch_size: int = 64,
    clustering: bool = False,
    workers: int = 0,
) -> float:
    pin_memory = device.type == "cuda"
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        persistent_workers=workers > 0,
        pin_memory=pin_memory,
    )
    embeddings = []
    labels = []
    mixed_precision = device.type in {"cuda", "mps"}
    model.eval()
    with torch.inference_mode():
        for base, batch_labels in loader:
            with torch.autocast(
                device.type,
                dtype=torch.float16,
                enabled=mixed_precision,
            ):
                batch_embeddings = model(base.to(device, non_blocking=pin_memory))
            embeddings.append(batch_embeddings.float().cpu())
            labels.extend(int(label) for label in batch_labels)
    embeddings = torch.cat(embeddings)
    if clustering:
        from aidetector.dazzlecow.metrics import clustering_metrics

        return clustering_metrics(
            embeddings.numpy(),
            labels,
            neighbors=neighbors,
        )["hungarian_accuracy"]
    return leave_one_out_knn_accuracy(embeddings, labels, neighbors)


def leave_one_out_knn_accuracy(
    embeddings: torch.Tensor,
    labels: list[int],
    neighbors: int = 5,
) -> float:
    if len(embeddings) < 2:
        return 0
    embeddings = torch.nn.functional.normalize(embeddings, dim=1)
    similarities = embeddings @ embeddings.T
    similarities.fill_diagonal_(float("-inf"))
    count = min(neighbors, len(embeddings) - 1)
    nearest = similarities.topk(count, dim=1).indices
    correct = 0
    for row, indices in enumerate(nearest):
        votes: dict[int, float] = {}
        for index in indices:
            label = labels[int(index)]
            votes[label] = votes.get(label, 0) + max(
                0,
                float(similarities[row, index]),
            )
        predicted = max(votes, key=votes.get)
        correct += predicted == labels[row]
    return correct / len(labels)


def build_gallery(
    model: Path,
    source: Path,
    output: Path,
    *,
    batch_size: int,
    device: str,
) -> None:
    samples = discover_public_dataset(f"identity={source}")
    encoder = DazzleCowEncoder(model, device=device)
    embeddings = []
    identities = []
    for offset in range(0, len(samples), batch_size):
        batch = samples[offset : offset + batch_size]
        images = [cv2.imread(str(sample.path)) for sample in batch]
        valid = [
            (sample, image)
            for sample, image in zip(batch, images, strict=True)
            if image is not None
        ]
        if not valid:
            continue
        embeddings.extend(encoder.embed([image for _, image in valid]))
        identities.extend(
            sample.identity.removeprefix("identity:") for sample, _ in valid
        )
    if not embeddings:
        raise ValueError(f"No readable gallery images found in {source}")
    save_gallery(output, np.asarray(embeddings, dtype=np.float32), identities)


def evaluate(
    model: Path,
    gallery_path: Path,
    source: Path,
    *,
    gallery_source: Path | None,
    failures_directory: Path | None,
    batch_size: int,
    neighbors: int,
    match_threshold: float,
    match_margin: float,
    device: str,
) -> dict[str, Any]:
    samples = discover_public_dataset(f"identity={source}")
    encoder = DazzleCowEncoder(model, device=device)
    gallery = DazzleCowGallery(
        gallery_path,
        neighbors=neighbors,
        match_threshold=match_threshold,
        match_margin=match_margin,
    )
    correct = 0
    matched = 0
    similarities = []
    confusion: Counter[tuple[str, str]] = Counter()
    totals: Counter[str] = Counter()
    correct_by_identity: Counter[str] = Counter()
    gallery_samples = None
    failure_rows = []
    if failures_directory is not None:
        if gallery_source is None:
            raise ValueError("gallery_source is required for failure reports")
        gallery_samples = [
            sample
            for sample in discover_public_dataset(f"identity={gallery_source}")
            if cv2.imread(str(sample.path)) is not None
        ]
        if len(gallery_samples) != len(gallery.identities):
            raise ValueError("Gallery source images do not match gallery embeddings")
        failures_directory.mkdir(parents=True, exist_ok=True)

    for offset in range(0, len(samples), batch_size):
        batch = samples[offset : offset + batch_size]
        images = [cv2.imread(str(sample.path)) for sample in batch]
        valid = [
            (sample, image)
            for sample, image in zip(batch, images, strict=True)
            if image is not None
        ]
        if not valid:
            continue
        embeddings = encoder.embed([image for _, image in valid])
        for (sample, image), embedding in zip(valid, embeddings, strict=True):
            expected = sample.identity.removeprefix("identity:")
            totals[expected] += 1
            result = gallery.match(embedding)
            if result is None:
                confusion[(expected, "unknown")] += 1
                continue
            matched += 1
            similarities.append(result.similarity)
            confusion[(expected, result.identity)] += 1
            if result.identity == expected:
                correct += 1
                correct_by_identity[expected] += 1
                continue

            if failures_directory is not None and gallery_samples is not None:
                matching_indices = np.flatnonzero(gallery.identities == result.identity)
                nearest = int(
                    matching_indices[
                        np.argmax(gallery.embeddings[matching_indices] @ embedding)
                    ]
                )
                gallery_sample = gallery_samples[nearest]
                filename = (
                    f"expected-{expected}_predicted-{result.identity}_"
                    f"{sample.path.stem}.jpg"
                )
                write_failure_image(
                    failures_directory / filename,
                    image,
                    cv2.imread(str(gallery_sample.path)),
                    expected,
                    result.identity,
                    result.similarity,
                )
                failure_rows.append(
                    (
                        sample.path,
                        expected,
                        result.identity,
                        result.similarity,
                        gallery_sample.path,
                    )
                )

    if failures_directory is not None:
        with open(failures_directory / "failures.csv", "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                ("query", "expected", "predicted", "similarity", "nearest_gallery")
            )
            writer.writerows(failure_rows)

    total = len(samples)
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
        "confusion": {
            f"{expected}->{predicted}": count
            for (expected, predicted), count in sorted(confusion.items())
        },
    }


def write_failure_image(
    path: Path,
    query: np.ndarray,
    gallery: np.ndarray,
    expected: str,
    predicted: str,
    similarity: float,
) -> None:
    size = 320
    header = 56
    image = np.full((size + header, size * 2, 3), 32, dtype=np.uint8)
    image[header:, :size] = fit_square(query, size)
    image[header:, size:] = fit_square(gallery, size)
    cv2.putText(
        image,
        f"Expected {expected}  Predicted {predicted}  Similarity {similarity:.3f}",
        (12, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "Query",
        (12, header + 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        image,
        "Nearest predicted gallery",
        (size + 12, header + 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    cv2.imwrite(str(path), image)


def fit_square(image: np.ndarray, size: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    resized = cv2.resize(
        image,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.full((size, size, 3), 32, dtype=np.uint8)
    y = (size - resized.shape[0]) // 2
    x = (size - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def train_main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the DazzleCow ResNet50 identity encoder"
    )
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        help="TYPE=PATH; TYPE is multicamcows2024, cows2021, or identity",
    )
    parser.add_argument(
        "--validation-dataset",
        action="append",
        help="Optional TYPE=PATH validation data used for best-checkpoint selection",
    )
    parser.add_argument("--output", type=Path, required=True)
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
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--architecture",
        choices=("resnet18", "resnet50"),
        default="resnet50",
    )
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, default=84000)
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    train(
        arguments.dataset,
        arguments.output,
        validation_specs=arguments.validation_dataset,
        epochs=arguments.epochs,
        identities_per_batch=arguments.identities_per_batch,
        images_per_identity=arguments.images_per_identity,
        workers=arguments.workers,
        learning_rate=arguments.learning_rate,
        temperature=arguments.temperature,
        patience=arguments.patience,
        training_mode=arguments.training_mode,
        device=arguments.device,
        seed=arguments.seed,
        architecture=arguments.architecture,
        image_size=arguments.image_size,
        resume=arguments.resume,
    )


def gallery_main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a kNN gallery from identity subdirectories"
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    arguments = parser.parse_args()
    build_gallery(
        arguments.model,
        arguments.source,
        arguments.output,
        batch_size=arguments.batch_size,
        device=arguments.device,
    )


def evaluate_main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a DazzleCow gallery on identity subdirectories"
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--gallery", type=Path, required=True)
    parser.add_argument("--gallery-source", type=Path)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--failures", type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--neighbors", type=int, default=5)
    parser.add_argument("--match-threshold", type=float, default=0)
    parser.add_argument("--match-margin", type=float, default=0)
    parser.add_argument("--device", default="auto")
    arguments = parser.parse_args()
    metrics = evaluate(
        arguments.model,
        arguments.gallery,
        arguments.source,
        gallery_source=arguments.gallery_source,
        failures_directory=arguments.failures,
        batch_size=arguments.batch_size,
        neighbors=arguments.neighbors,
        match_threshold=arguments.match_threshold,
        match_margin=arguments.match_margin,
        device=arguments.device,
    )
    print(
        f"Top-1 accuracy: {metrics['accuracy']:.1%} "
        f"({metrics['correct']}/{metrics['total']}), "
        f"coverage: {metrics['coverage']:.1%}, "
        f"matched accuracy: {metrics['matched_accuracy']:.1%}, "
        f"mean similarity: {metrics['mean_similarity']:.3f}"
    )
    print(
        "Per identity: "
        + ", ".join(
            f"{identity}={accuracy:.1%}"
            for identity, accuracy in metrics["per_identity"].items()
        )
    )
