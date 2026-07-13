from dataclasses import dataclass, replace
from typing import Any

import cv2
import numpy as np
from aidetector.domain.detections import DetectedObject
from aidetector.domain.identity import IdentityCandidate
from numpy import ndarray


@dataclass(frozen=True)
class LocalizerSettings:
    label: str
    confidence: float
    min_area_ratio: float
    max_area_ratio: float
    margin: float


class SegmentationLocalizer:
    def __init__(self, settings: LocalizerSettings):
        self.settings = settings

    def candidates(self, result: Any, frame: ndarray) -> list[IdentityCandidate]:
        return candidates_from_result(result, frame, self.settings)


def candidates_from_result(
    result: Any,
    frame: ndarray,
    settings: LocalizerSettings,
) -> list[IdentityCandidate]:
    masks = result.masks
    boxes = result.boxes
    if masks is None or boxes is None:
        return []

    background = None
    candidates = []
    for box, raw_mask in zip(boxes, masks.data, strict=True):
        class_id = int(box.cls.item())
        if result.names[class_id] != settings.label:
            continue
        confidence = float(box.conf.item())
        if confidence < settings.confidence:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        if not box_allowed(
            (x1, y1, x2, y2),
            frame.shape[:2],
            settings.min_area_ratio,
            settings.max_area_ratio,
            settings.margin,
        ):
            continue

        mask = raw_mask.detach().cpu().numpy().astype(bool)
        if mask.shape != frame.shape[:2]:
            mask = cv2.resize(
                mask.astype(np.uint8),
                (frame.shape[1], frame.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        if background is None:
            background = (
                frame.reshape(-1, frame.shape[2]).mean(axis=0).astype(frame.dtype)
            )
        candidate = masked_candidate(
            frame,
            mask,
            settings.label,
            confidence,
            background,
        )
        if candidate is not None:
            candidates.append(
                replace(
                    candidate,
                    detection=replace(
                        candidate.detection,
                        track_id=_track_id(box),
                    ),
                )
            )
    return candidates


def box_allowed(
    box: tuple[int, int, int, int],
    image_shape: tuple[int, int],
    min_area_ratio: float,
    max_area_ratio: float,
    margin: float = 0,
) -> bool:
    height, width = image_shape
    x1, y1, x2, y2 = box
    area_ratio = (x2 - x1) * (y2 - y1) / (width * height)
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return (
        min_area_ratio <= area_ratio <= max_area_ratio
        and width * margin <= center_x <= width * (1 - margin)
        and height * margin <= center_y <= height * (1 - margin)
    )


def masked_candidate(
    frame: ndarray,
    mask: ndarray,
    label: str,
    confidence: float,
    background: ndarray | None = None,
) -> IdentityCandidate | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None

    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max()) + 1
    y2 = int(ys.max()) + 1
    source = frame[y1:y2, x1:x2]
    crop_mask = mask[y1:y2, x1:x2]
    if background is None:
        background = frame.reshape(-1, frame.shape[2]).mean(axis=0).astype(frame.dtype)
    masked = np.empty_like(source)
    masked[:] = background
    masked[crop_mask] = source[crop_mask]
    return IdentityCandidate(
        DetectedObject(x1, y1, x2, y2, label=label, confidence=confidence),
        masked,
    )


def _track_id(box: Any) -> int | None:
    value = box.id
    return int(value.item()) if value is not None else None
