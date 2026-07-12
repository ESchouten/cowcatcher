from collections.abc import Sequence

import cv2
import numpy as np

from aidetector.domain.detections import DetectedObject, Observation
from aidetector.domain.frames import Frame


def even_width(value: int) -> int:
    return max(2, value // 2 * 2)


def get_image(image: np.ndarray, quality: int = 100) -> bytes:
    success, jpg = cv2.imencode(
        ".jpg",
        image,
        (int(cv2.IMWRITE_JPEG_QUALITY), quality),
    )
    if not success:
        raise ValueError("Failed to encode image")
    return jpg.tobytes()


def frame_image(frame: Frame) -> np.ndarray:
    if frame.image is not None:
        return frame.image
    assert frame.jpeg is not None
    image = cv2.imdecode(np.frombuffer(frame.jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode frame JPEG")
    return image


def frame_jpg(frame: Frame, quality: int = 100) -> bytes:
    if frame.jpeg is not None:
        return frame.jpeg
    return get_image(frame.require_image(), quality)


def get_plot(
    observation: Observation,
    objects: Sequence[DetectedObject] | None = None,
) -> np.ndarray:
    objects = objects if objects is not None else observation.objects
    if not objects:
        return frame_image(observation.frame)

    image = frame_image(observation.frame).copy()
    height, width = image.shape[:2]
    color = (255, 0, 0)
    thickness = max(2, round(min(width, height) / 500))
    font_scale = max(0.5, min(width, height) / 1200)
    font_thickness = max(1, round(thickness / 2))

    for item in objects:
        x1 = max(0, min(width - 1, item.x1))
        y1 = max(0, min(height - 1, item.y1))
        x2 = max(0, min(width - 1, item.x2))
        y2 = max(0, min(height - 1, item.y2))
        if x2 <= x1 or y2 <= y1:
            continue

        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        if item.label is None or item.confidence is None:
            continue

        label = f"{item.label} {item.confidence:.0%}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            font_thickness,
        )
        label_y1 = max(0, y1 - text_height - baseline - thickness)
        label_y2 = label_y1 + text_height + baseline + thickness
        label_x2 = min(width - 1, x1 + text_width + thickness * 2)
        cv2.rectangle(image, (x1, label_y1), (label_x2, label_y2), color, -1)
        cv2.putText(
            image,
            label,
            (x1 + thickness, label_y2 - baseline - max(1, thickness // 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            font_thickness,
            cv2.LINE_AA,
        )
    return image


def shrink_image(image: np.ndarray, width_max: int) -> np.ndarray:
    height, width = image.shape[:2]
    width_max = even_width(width_max)
    if width <= width_max:
        return image

    scale = width_max / width
    new_height = even_width(round(height * scale))
    return cv2.resize(
        image,
        (width_max, new_height),
        interpolation=cv2.INTER_AREA,
    )


def compress_jpg(
    image: np.ndarray,
    max_bytes: int,
    start_quality: int = 90,
    min_quality: int = 10,
    min_scale: float = 0.1,
    quality_step: int = 10,
    scale_step: float = 0.9,
) -> bytes:
    quality = start_quality
    jpg = get_image(image, quality)

    while len(jpg) > max_bytes and quality > min_quality:
        quality = max(min_quality, quality - quality_step)
        jpg = get_image(image, quality)

    scale = 1.0
    while len(jpg) > max_bytes and scale > min_scale:
        scale = max(min_scale, scale * scale_step)
        width = max(1, int(image.shape[1] * scale))
        height = max(1, int(image.shape[0] * scale))
        resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        jpg = get_image(resized, quality)

    return jpg


def get_crop(
    observation: Observation,
    crop: DetectedObject | None = None,
    aspect_ratio: float | None = 16 / 9,
    padding: float = 0.1,
    plot: bool = True,
    plot_objects: Sequence[DetectedObject] | None = None,
) -> np.ndarray | None:
    crop = crop or observation.crop_region
    if crop is None:
        return None
    image = (
        get_plot(observation, plot_objects) if plot else frame_image(observation.frame)
    )
    height, width = image.shape[:2]
    box_width = max(1, crop.x2 - crop.x1)
    box_height = max(1, crop.y2 - crop.y1)
    pad_x, pad_y = int(box_width * padding), int(box_height * padding)
    x1, y1 = max(0, crop.x1 - pad_x), max(0, crop.y1 - pad_y)
    x2, y2 = min(width, crop.x2 + pad_x), min(height, crop.y2 + pad_y)

    if x2 <= x1 or y2 <= y1:
        return None

    if aspect_ratio and aspect_ratio > 0:
        crop_width = x2 - x1
        crop_height = y2 - y1
        target_width = crop_width
        target_height = crop_height
        current_ratio = crop_width / crop_height
        if current_ratio < aspect_ratio:
            target_width = int(round(crop_height * aspect_ratio))
        elif current_ratio > aspect_ratio:
            target_height = int(round(crop_width / aspect_ratio))

        if target_width > width:
            target_width = width
            target_height = int(round(target_width / aspect_ratio))
        if target_height > height:
            target_height = height
            target_width = int(round(target_height * aspect_ratio))

        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        x1, x2 = _centered_range(center_x, target_width, width)
        y1, y2 = _centered_range(center_y, target_height, height)

    return image[y1:y2, x1:x2]


def _centered_range(center: float, size: int, limit: int) -> tuple[int, int]:
    size = max(1, min(size, limit))
    start = int(round(center - size / 2))
    end = start + size
    if start < 0:
        end -= start
        start = 0
    if end > limit:
        start -= end - limit
        end = limit
    return max(0, start), min(limit, end)
