from dataclasses import dataclass
from datetime import datetime
from typing import Any

import cv2
import numpy as np
from aidetector.detection.yolo import TrackedSourceResult, YoloRunner
from aidetector.utils.config import (
    CowIdentityConfig,
    Crop,
    OnnxConfig,
    YoloConfig,
)
from numpy import ndarray
from PIL import Image


@dataclass
class CowCandidate:
    crop: Crop
    image: ndarray


@dataclass(frozen=True)
class LocalizerSettings:
    model: str
    confidence: float
    min_area_ratio: float
    max_area_ratio: float
    margin: float
    nms_iou: float
    imgsz: int

    @classmethod
    def from_config(cls, config: CowIdentityConfig) -> "LocalizerSettings":
        return cls(
            config.segment_model,
            config.confidence,
            config.min_area_ratio,
            config.max_area_ratio,
            config.margin,
            config.nms_iou,
            config.imgsz,
        )


class DazzleCowLocalizer:
    def __init__(
        self,
        settings: LocalizerSettings,
        onnx_config: OnnxConfig | None = None,
        sources: list[str] | None = None,
    ):
        self.settings = settings
        self.runner = YoloRunner(
            YoloConfig(
                model=settings.model,
                task="segment",
                tracking=True,
                confidence={"cow": settings.confidence},
                imgsz=settings.imgsz,
                iou=settings.nms_iou,
                tracker="bytetrack.yaml",
            ),
            onnx_config or OnnxConfig(),
            sources or ["source"],
        )

    def locate(self, frame: ndarray) -> list[CowCandidate]:
        result = self.runner.detect([frame])[0]
        return candidates_from_result(result, frame, self.settings)

    def track_sources(
        self,
        batch: dict[str, list[tuple[datetime, ndarray]]],
    ) -> list[TrackedSourceResult]:
        return [
            TrackedSourceResult(
                tracked.source,
                candidates_from_result(
                    tracked.result,
                    tracked.frames[-1][1],
                    self.settings,
                ),
                tracked.frames,
            )
            for tracked in self.runner.track_sources(batch)
        ]


def candidates_from_result(
    result: Any,
    frame: ndarray,
    settings: LocalizerSettings,
) -> list[CowCandidate]:
    masks = getattr(result, "masks", None)
    boxes = getattr(result, "boxes", None)
    if masks is None or boxes is None:
        return []

    candidates = []
    for box, raw_mask in zip(boxes, masks.data, strict=False):
        if _box_label(result, box) not in (None, "cow"):
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
        candidate = masked_candidate(frame, mask, float(box.conf.item()))
        if candidate is not None:
            candidate.crop.track_id = _track_id(box)
            candidates.append(candidate)
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
    area_ratio = max(0, x2 - x1) * max(0, y2 - y1) / max(1, width * height)
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
    confidence: float,
) -> CowCandidate | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None

    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max()) + 1
    y2 = int(ys.max()) + 1
    background = frame.reshape(-1, frame.shape[2]).mean(axis=0).astype(frame.dtype)
    masked = np.empty_like(frame)
    masked[:] = background
    masked[mask] = frame[mask]
    return CowCandidate(
        Crop(x1, y1, x2, y2, label="cow", confidence=confidence),
        masked[y1:y2, x1:x2],
    )


def segment_candidates(
    sam,
    frame: ndarray,
    boxes: list[list[int]],
    scores: list[float],
    *,
    device: str = "auto",
) -> list[CowCandidate]:
    """Mask known boxes for dataset preparation and oracle benchmarks."""
    if not boxes:
        return []

    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    points = [[(box[0] + box[2]) // 2, (box[1] + box[3]) // 2] for box in boxes]
    result = sam.predict(
        source=np.asarray(image),
        bboxes=boxes,
        points=points,
        labels=np.ones(len(boxes), dtype=np.int32),
        device=None if device == "auto" else device,
        verbose=False,
    )[0]
    masks = getattr(result, "masks", None)
    if masks is None:
        return []

    candidates = []
    for index, raw_mask in enumerate(masks.data):
        mask = raw_mask.detach().cpu().numpy().astype(bool)
        if mask.shape != frame.shape[:2]:
            mask = cv2.resize(
                mask.astype(np.uint8),
                (frame.shape[1], frame.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        candidate = masked_candidate(
            frame,
            mask,
            scores[index] if index < len(scores) else 1.0,
        )
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def _track_id(box: Any) -> int | None:
    value = getattr(box, "id", None)
    return int(value.item()) if value is not None else None


def _box_label(result: Any, box: Any) -> str | None:
    names = getattr(result, "names", None)
    class_value = getattr(box, "cls", None)
    if names is None or class_value is None:
        return None
    class_id = int(class_value.item())
    if isinstance(names, dict):
        return str(names.get(class_id)) if class_id in names else None
    return str(names[class_id]) if 0 <= class_id < len(names) else None
