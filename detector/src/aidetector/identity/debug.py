import logging
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from aidetector.utils.config import Crop, DetectedObject


def save_identity_debug(
    debug_directory: Path | None,
    image: np.ndarray,
    objects: list[DetectedObject],
    selected: list[DetectedObject],
    source: str,
    crop_label: str | None,
    identity_images: list[np.ndarray] | None = None,
    target_crop: Crop | None = None,
) -> None:
    if debug_directory is None:
        return

    try:
        debug_directory.mkdir(parents=True, exist_ok=True)
        overlay = image.copy()
        if target_crop is not None:
            cv2.rectangle(
                overlay,
                (target_crop.x1, target_crop.y1),
                (target_crop.x2, target_crop.y2),
                (255, 0, 0),
                2,
            )
            center_x = int((target_crop.x1 + target_crop.x2) / 2)
            center_y = int((target_crop.y1 + target_crop.y2) / 2)
            cv2.drawMarker(
                overlay,
                (center_x, center_y),
                (255, 0, 0),
                cv2.MARKER_CROSS,
                18,
                2,
            )
        selected_ids = {id(obj) for obj in selected}
        for obj in objects:
            color = (0, 180, 255) if id(obj) in selected_ids else (0, 0, 255)
            if obj.mask is not None:
                mask_layer = np.zeros_like(overlay)
                mask_layer[obj.mask] = color
                overlay = cv2.addWeighted(overlay, 1.0, mask_layer, 0.35, 0)
            crop = obj.crop
            cv2.rectangle(overlay, (crop.x1, crop.y1), (crop.x2, crop.y2), color, 2)
            cv2.putText(
                overlay,
                _object_label(crop.label, crop.confidence),
                (crop.x1, max(crop.y1 - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

        filename = (
            f"{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}-"
            f"{_safe_filename(source)}-{_safe_filename(crop_label or 'unknown')}-"
            f"{'selected' if selected else 'failed'}.jpg"
        )
        cv2.imwrite(str(debug_directory / filename), overlay)
        for index, identity_image in enumerate(identity_images or [], start=1):
            suffix = (
                "-megadescriptor.png"
                if len(identity_images or []) == 1
                else f"-megadescriptor-{index}.png"
            )
            cv2.imwrite(
                str(debug_directory / filename.replace(".jpg", suffix)),
                identity_image,
            )
    except Exception:
        logging.getLogger(__name__).exception("Failed to write identity debug image")


def format_detected_labels(objects: list[DetectedObject]) -> str:
    if not objects:
        return "[]"

    labels = [_object_label(obj.crop.label, obj.crop.confidence) for obj in objects]
    return "[" + ", ".join(labels) + "]"


def _object_label(label: str | None, confidence: float | None) -> str:
    if confidence is None:
        return label or "unknown"
    return f"{label or 'unknown'} {confidence:.2f}"


def _safe_filename(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_." else "_" for char in value)[
        :80
    ]
