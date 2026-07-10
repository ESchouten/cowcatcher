import argparse
import logging
from pathlib import Path

import cv2
from aidetector.dazzlecow.localizer import (
    CowCandidate,
    DazzleCowLocalizer,
    LocalizerSettings,
)
from aidetector.dazzlecow.train import discover_public_dataset

logger = logging.getLogger(__name__)


def prepare(
    dataset_specs: list[str],
    output: Path,
    settings: LocalizerSettings,
) -> None:
    localizer = DazzleCowLocalizer(settings)
    samples = [
        sample
        for specification in dataset_specs
        for sample in discover_public_dataset(specification)
    ]
    output.mkdir(parents=True, exist_ok=True)
    written = 0
    for index, sample in enumerate(samples, start=1):
        frame = cv2.imread(str(sample.path))
        if frame is None:
            logger.warning("Skipping unreadable image %s", sample.path)
            continue
        candidates = localizer.locate(frame)
        candidate = centered_candidate(candidates, frame.shape[:2])
        if candidate is None:
            logger.warning("No cow mask found in %s", sample.path)
            continue

        identity = sample.identity.replace(":", "__")
        group = (sample.group or "default").replace("/", "_")
        timestamp = (sample.timestamp or sample.path.stem).replace("/", "_")
        target = output / identity / f"{group}--{timestamp}--{index:08d}.jpg"
        target.parent.mkdir(parents=True, exist_ok=True)
        if cv2.imwrite(str(target), candidate.image):
            written += 1
        if index % 100 == 0:
            logger.info("Prepared %d/%d images", index, len(samples))
    logger.info("Prepared %d of %d images in %s", written, len(samples), output)


def centered_candidate(
    candidates: list[CowCandidate],
    image_shape: tuple[int, int],
) -> CowCandidate | None:
    if not candidates:
        return None
    height, width = image_shape
    center_x, center_y = width / 2, height / 2
    return min(
        candidates,
        key=lambda candidate: (
            ((candidate.crop.x1 + candidate.crop.x2) / 2 - center_x) ** 2
            + ((candidate.crop.y1 + candidate.crop.y2) / 2 - center_y) ** 2,
            -(candidate.crop.confidence or 0),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create paper-style OWLv2 + SAM2 masked public cow data"
    )
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--owl-model",
        default="google/owlv2-large-patch14-ensemble",
    )
    parser.add_argument("--sam-model", default="sam2.1_l.pt")
    parser.add_argument("--prompt", default="cow")
    parser.add_argument("--confidence", type=float, default=0.3)
    parser.add_argument("--min-area-ratio", type=float, default=0.01)
    parser.add_argument("--max-area-ratio", type=float, default=1.0)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    prepare(
        arguments.dataset,
        arguments.output,
        LocalizerSettings(
            arguments.owl_model,
            arguments.sam_model,
            arguments.prompt,
            arguments.confidence,
            arguments.min_area_ratio,
            arguments.max_area_ratio,
            arguments.nms_iou,
            arguments.device,
        ),
    )
