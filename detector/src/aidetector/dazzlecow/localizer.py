from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import cv2
import numpy as np
from aidetector.dazzlecow.geometry import box_iou
from aidetector.utils.config import Crop, DazzleCowConfig
from numpy import ndarray
from PIL import Image
from ultralytics.data.loaders import LoadStreams, SourceTypes


@dataclass
class CowCandidate:
    crop: Crop
    image: ndarray


@dataclass(frozen=True)
class LocalizerSettings:
    owl_model: str
    sam_model: str
    prompt: str
    confidence: float
    min_area_ratio: float
    max_area_ratio: float
    nms_iou: float
    device: str

    @classmethod
    def from_config(cls, config: DazzleCowConfig) -> "LocalizerSettings":
        return cls(
            config.owl_model,
            config.sam_model,
            config.prompt,
            config.confidence,
            config.min_area_ratio,
            config.max_area_ratio,
            config.nms_iou,
            config.device,
        )


class DazzleCowLocalizer:
    def __init__(self, settings: LocalizerSettings):
        try:
            from transformers import pipeline
        except ImportError as error:
            raise RuntimeError(
                "DazzleCow requires the 'dazzlecow' optional dependencies"
            ) from error

        from ultralytics import SAM

        self.settings = settings
        self.owl = pipeline(
            task="zero-shot-object-detection",
            model=settings.owl_model,
            device=_transformers_device(settings.device),
        )
        self.sam = SAM(settings.sam_model)

    def locate(self, frame: ndarray) -> list[CowCandidate]:
        boxes, scores = self.boxes(frame)
        return segment_candidates(
            self.sam,
            frame,
            boxes,
            scores,
            device=self.settings.device,
        )

    def boxes(self, frame: ndarray) -> tuple[list[list[int]], list[float]]:
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        outputs = self.owl(
            image,
            candidate_labels=[self.settings.prompt],
            threshold=self.settings.confidence,
        )
        boxes, scores = filtered_boxes(
            outputs,
            frame.shape[:2],
            self.settings.min_area_ratio,
            self.settings.max_area_ratio,
            self.settings.nms_iou,
        )
        return boxes, scores


class DazzleCowVideoLocalizer(DazzleCowLocalizer):
    def __init__(self, settings: LocalizerSettings, owl_interval: float):
        super().__init__(settings)
        from ultralytics.models.sam import SAM2VideoPredictor

        self.owl_interval = timedelta(seconds=max(0, owl_interval))
        self.last_owl: dict[str, datetime] = {}
        self.states: dict[str, dict] = {}
        self.frames: dict[str, int] = {}
        self.scores: dict[str, list[float]] = {}
        self.predictor = SAM2VideoPredictor(
            overrides={
                "task": "segment",
                "mode": "predict",
                "model": settings.sam_model,
                "imgsz": 1024,
                "device": None if settings.device == "auto" else settings.device,
                "verbose": False,
                "save": False,
            },
            _callbacks=self.sam.callbacks,
        )
        self.predictor.setup_model(model=self.sam.model, verbose=False)

    def locate(
        self,
        source: str,
        frame: ndarray,
        date: datetime,
    ) -> list[CowCandidate]:
        boxes = None
        should_relocate = (
            source not in self.states
            or date - self.last_owl.get(source, datetime.min) >= self.owl_interval
        )
        if should_relocate:
            detected_boxes, scores = self.boxes(frame)
            self.last_owl[source] = date
            if detected_boxes:
                boxes = detected_boxes
                self.scores[source] = scores
                self.states.pop(source, None)
                self.frames[source] = 0

        state = self.states.get(source)
        if state is None and boxes is None:
            return []

        frame_index = self.frames.get(source, 0)
        self.predictor.inference_state = state or {}
        results = list(
            self.predictor(
                source=_SAMVideoFrame(source, frame, frame_index),
                bboxes=boxes,
                stream=True,
            )
        )
        self.states[source] = self.predictor.inference_state
        self.frames[source] = frame_index + 1
        if not results or results[0].masks is None:
            return []
        return candidates_from_masks(
            frame,
            results[0].masks.data,
            self.scores.get(source, []),
        )


class _SAMVideoFrame(LoadStreams):
    def __init__(self, source: str, image: ndarray, frame: int):
        self.source = source
        self.image = image
        self.frame = frame
        self.frames = 2**31
        self.bs = 1
        self.mode = "video"
        self.video_flag = [True]
        self.source_type = SourceTypes(stream=True)
        self.count = 0

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self) -> tuple[list[str], list[ndarray], list[str]]:
        if self.count:
            raise StopIteration
        self.count += 1
        return [self.source], [self.image], [""]

    def __len__(self) -> int:
        return 1

    def close(self) -> None:
        pass


def segment_candidates(
    sam,
    frame: ndarray,
    boxes: list[list[int]],
    scores: list[float],
    *,
    device: str = "auto",
) -> list[CowCandidate]:
    if not boxes:
        return []

    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    points = [
        [(box[0] + box[2]) // 2, (box[1] + box[3]) // 2]
        for box in boxes
    ]
    result = sam.predict(
        source=np.asarray(image),
        bboxes=boxes,
        points=points,
        labels=np.ones(len(boxes), dtype=np.int32),
        device=None if device == "auto" else device,
        verbose=False,
    )[0]
    masks = getattr(result, "masks", None)
    return candidates_from_masks(frame, masks.data if masks is not None else [], scores)


def candidates_from_masks(
    frame: ndarray,
    masks,
    scores: list[float],
) -> list[CowCandidate]:
    candidates = []
    for index, raw_mask in enumerate(masks):
        mask = raw_mask.detach().cpu().numpy().astype(bool)
        if mask.shape != frame.shape[:2]:
            mask = cv2.resize(
                mask.astype(np.uint8),
                (frame.shape[1], frame.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        score = scores[index] if index < len(scores) else 1.0
        candidate = masked_candidate(frame, mask, score)
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def filtered_boxes(
    outputs: list[dict[str, Any]],
    image_shape: tuple[int, int],
    min_area_ratio: float,
    max_area_ratio: float,
    nms_iou: float,
) -> tuple[list[list[int]], list[float]]:
    height, width = image_shape
    candidates = []
    for output in outputs:
        box = output["box"]
        coords = [
            int(box["xmin"]),
            int(box["ymin"]),
            int(box["xmax"]),
            int(box["ymax"]),
        ]
        area = max(0, coords[2] - coords[0]) * max(0, coords[3] - coords[1])
        ratio = area / max(1, width * height)
        if min_area_ratio <= ratio <= max_area_ratio:
            candidates.append((coords, float(output["score"])))

    candidates.sort(key=lambda item: item[1], reverse=True)
    kept: list[tuple[list[int], float]] = []
    for box, score in candidates:
        if all(box_iou(box, other) <= nms_iou for other, _ in kept):
            kept.append((box, score))
    return [box for box, _ in kept], [score for _, score in kept]


def masked_candidate(frame: ndarray, mask: ndarray, confidence: float) -> CowCandidate | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None

    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    background = frame.reshape(-1, frame.shape[2]).mean(axis=0).astype(frame.dtype)
    masked = np.empty_like(frame)
    masked[:] = background
    masked[mask] = frame[mask]
    return CowCandidate(
        Crop(x1, y1, x2, y2, label="cow", confidence=confidence),
        masked[y1:y2, x1:x2],
    )
def _transformers_device(device: str):
    if device == "auto":
        import torch

        if torch.cuda.is_available():
            return 0
        if torch.backends.mps.is_available():
            return "mps"
        return -1
    return device
