from dataclasses import dataclass

import cv2
import numpy as np
from aidetector.utils.config import Crop


@dataclass
class YoloObject:
    crop: Crop
    mask: np.ndarray | None = None

    @property
    def area(self) -> int:
        if self.mask is not None:
            return int(self.mask.sum())
        return max(0, self.crop.x2 - self.crop.x1) * max(0, self.crop.y2 - self.crop.y1)


def objects_from_result(
    result,
    image_shape: tuple[int, int],
    class_confidences: dict[int, tuple[str, float]] | None = None,
    labels: list[str] | None = None,
    min_confidence: float = 0,
) -> list[YoloObject]:
    if result.boxes is None:
        return []

    names = result.names or {}
    accepted_labels = set(labels) if labels is not None else None
    objects: list[YoloObject] = []

    for index, box in enumerate(result.boxes):
        class_id = int(box.cls.item())
        class_name = _class_name(names, class_id)
        confidence = float(box.conf.item())
        threshold = min_confidence

        if class_confidences is not None:
            if class_id not in class_confidences:
                continue
            class_name, threshold = class_confidences[class_id]

        if accepted_labels is not None and class_name not in accepted_labels:
            continue
        if confidence < threshold:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        mask = _mask_from_result(result, index, image_shape)
        objects.append(
            YoloObject(
                crop=Crop(
                    x1,
                    y1,
                    x2,
                    y2,
                    label=class_name,
                    confidence=confidence,
                ),
                mask=mask,
            )
        )

    return objects


def confidences_from_objects(objects: list[YoloObject]) -> dict[str, float]:
    confidences: dict[str, float] = {}
    for obj in objects:
        crop = obj.crop
        if crop.label is None or crop.confidence is None:
            continue
        confidences[crop.label] = max(
            confidences.get(crop.label, 0),
            crop.confidence,
        )
    return confidences


def apply_mask(
    image: np.ndarray,
    mask: np.ndarray,
    background: str,
) -> np.ndarray:
    background_value = 0 if background == "black" else 127
    masked = np.full_like(image, background_value)
    masked[mask] = image[mask]
    return masked


def _mask_from_result(result, index: int, image_shape: tuple[int, int]) -> np.ndarray | None:
    if result.masks is None or index >= len(result.masks.data):
        return None

    mask = result.masks.data[index].detach().cpu().numpy().astype(bool)
    if mask.shape == image_shape:
        return mask

    return cv2.resize(
        mask.astype(np.uint8),
        (image_shape[1], image_shape[0]),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)


def _class_name(names, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)
