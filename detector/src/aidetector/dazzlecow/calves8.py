import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from aidetector.dazzlecow.localizer import segment_candidates

logger = logging.getLogger(__name__)
SPLITS = (
    ("train", 0.0, 0.5, 0.6),
    ("validation", 0.55, 0.65, 0.15),
    ("gallery", 0.7, 0.8, 0.1),
    ("test", 0.85, 1.0, 0.15),
)


def select_annotations(
    annotations: pd.DataFrame,
    samples_per_identity: int,
) -> pd.DataFrame:
    if samples_per_identity < len(SPLITS):
        raise ValueError(f"samples_per_identity must be at least {len(SPLITS)}")

    selected = []
    frame_max = int(annotations["frame_id"].max())
    for _, identity_rows in annotations.groupby("tracklet_id"):
        identity_rows = identity_rows.sort_values("frame_id").drop_duplicates("frame_id")
        allocated = 0
        for index, (split, start, end, fraction) in enumerate(SPLITS):
            count = (
                samples_per_identity - allocated
                if index == len(SPLITS) - 1
                else round(samples_per_identity * fraction)
            )
            allocated += count
            candidates = identity_rows[
                (identity_rows["frame_id"] > int(frame_max * start))
                & (identity_rows["frame_id"] <= int(frame_max * end))
            ]
            if candidates.empty:
                continue
            indices = np.linspace(
                0,
                len(candidates) - 1,
                min(count, len(candidates)),
                dtype=int,
            )
            rows = candidates.iloc[indices].copy()
            rows["split"] = split
            selected.append(rows)

    if not selected:
        raise ValueError("No 8-calves annotations selected")
    return pd.concat(selected, ignore_index=True).sort_values("frame_id")


def normalized_box(row, width: int, height: int) -> list[int]:
    x1 = max(0, round((row.x - row.w / 2) * width))
    y1 = max(0, round((row.y - row.h / 2) * height))
    x2 = min(width, round((row.x + row.w / 2) * width))
    y2 = min(height, round((row.y + row.h / 2) * height))
    return [x1, y1, x2, y2]


def prepare_calves8(
    video: Path,
    annotations_path: Path,
    output: Path,
    *,
    samples_per_identity: int,
    sam_model: str,
    device: str,
) -> None:
    from ultralytics import SAM

    annotations = pd.read_pickle(annotations_path)
    selected = select_annotations(annotations, samples_per_identity)
    sam = SAM(sam_model)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    written = 0
    try:
        for frame_id, rows in selected.groupby("frame_id", sort=True):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id) - 1)
            success, frame = capture.read()
            if not success:
                logger.warning("Could not read frame %s", frame_id)
                continue

            boxes = [normalized_box(row, width, height) for row in rows.itertuples()]
            scores = [float(row.conf) for row in rows.itertuples()]
            candidates = segment_candidates(
                sam,
                frame,
                boxes,
                scores,
                device=device,
            )
            if len(candidates) != len(rows):
                logger.warning(
                    "SAM returned %d of %d masks for frame %s",
                    len(candidates),
                    len(rows),
                    frame_id,
                )
                continue

            for row, candidate in zip(rows.itertuples(), candidates, strict=True):
                target = (
                    output
                    / row.split
                    / f"{int(row.tracklet_id):03d}"
                    / f"{int(frame_id):06d}.jpg"
                )
                target.parent.mkdir(parents=True, exist_ok=True)
                if cv2.imwrite(str(target), candidate.image):
                    written += 1
            if written and written % 100 == 0:
                logger.info("Prepared %d masked 8-calves images", written)
    finally:
        capture.release()

    logger.info("Prepared %d images in %s", written, output)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare chronological masked train/validation/gallery/test crops from 8-calves"
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-identity", type=int, default=40)
    parser.add_argument("--sam-model", default="sam2.1_l.pt")
    parser.add_argument("--device", default="auto")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    prepare_calves8(
        arguments.video,
        arguments.annotations,
        arguments.output,
        samples_per_identity=arguments.samples_per_identity,
        sam_model=arguments.sam_model,
        device=arguments.device,
    )
