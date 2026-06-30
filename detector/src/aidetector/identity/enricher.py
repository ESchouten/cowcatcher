import logging
from dataclasses import dataclass
from datetime import datetime
from threading import Lock

import cv2
import numpy as np
from aidetector.detection.yolo import apply_mask, objects_from_result
from aidetector.identity.service import IdentityService
from aidetector.utils.config import (
    Crop,
    DetectedObject,
    Detection,
    DetectorIdentityConfig,
    IdentityFallbackConfig,
)


class IdentityEnricher:
    logger = logging.getLogger(__name__)

    def __init__(self, service: IdentityService, config: DetectorIdentityConfig):
        self.service = service
        self.config = config
        self.fallback_model = None
        self.fallback_lock = Lock()

    def enrich(
        self,
        source: str,
        detection: Detection,
    ) -> None:
        detection.identities = []
        for obj in detection.images.objects:
            obj.identity = None

        lookups = self._identity_lookups(detection, source)
        if not lookups:
            self.logger.info("Identity provider received no lookup images")
            return

        self.logger.info(
            "Looking up %d identity image(s) with provider %s",
            len(lookups),
            self.config.provider,
        )
        identities = self.service.identify(
            self.config.provider,
            [lookup.image for lookup in lookups],
            source,
        )
        if not identities:
            self.logger.info("Identity provider returned no result")
            return

        for lookup, identity in zip(lookups, identities):
            if lookup.target.identity is None:
                lookup.target.identity = identity

        primary = identities[0]
        self.logger.info(
            "Identity result: status=%s id=%s similarity=%s",
            primary.status,
            primary.identity_id,
            primary.similarity,
        )
        detection.identities = identities

    def _identity_lookups(
        self,
        detection: Detection,
        source: str,
    ) -> list["IdentityLookup"]:
        objects = [
            obj
            for obj in detection.images.objects
            if _label_matches(obj.crop.label, self.config.labels)
        ]
        if not objects:
            self.logger.info("Skipping identity: no detector objects matched")
            return []

        masked_objects = [obj for obj in objects if obj.mask is not None]
        if masked_objects:
            return self._object_lookups(
                detection.images.jpg,
                masked_objects,
                source,
                debug_label="detector",
            )

        if self.config.fallback is not None:
            return self._fallback_identity_lookups(
                detection,
                source,
                self.config.fallback,
                objects[0],
            )

        return self._object_lookups(
            detection.images.jpg,
            objects,
            source,
            debug_label="detector",
        )

    def _object_lookups(
        self,
        image: np.ndarray,
        objects: list[DetectedObject],
        source: str,
        debug_label: str | None,
    ) -> list["IdentityLookup"]:
        selected = objects if self.config.multiple else objects[:1]
        lookups = [
            IdentityLookup(target=obj, image=identity_image)
            for obj in selected
            if (identity_image := _object_identity_image(image, obj)) is not None
        ]
        _save_identity_debug(
            self.config,
            image,
            objects,
            selected,
            source,
            debug_label,
            [lookup.image for lookup in lookups],
        )
        return lookups

    def _fallback_identity_lookups(
        self,
        detection: Detection,
        source: str,
        config: IdentityFallbackConfig,
        target: DetectedObject,
    ) -> list["IdentityLookup"]:
        image = detection.images.jpg
        crop = target.crop

        detected, matched, selected = self._fallback_objects(image, config, crop)
        if not matched:
            self.logger.info(
                "Identity fallback model %s found labels %s, but none matched %s with masks",
                config.model,
                _format_detected_labels(detected),
                config.labels,
            )
            _save_identity_debug(
                self.config,
                image,
                detected,
                [],
                source,
                crop.label,
                target_crop=crop,
            )
            return []

        if not selected:
            self.logger.info(
                "Skipping identity fallback: %s found %s, but none were centered in %s crop",
                config.model,
                _format_detected_labels(matched),
                crop.label,
            )
            _save_identity_debug(
                self.config,
                image,
                matched,
                [],
                source,
                crop.label,
                target_crop=crop,
            )
            return []

        lookups: list[IdentityLookup] = []
        for obj in selected:
            identity_image = _object_identity_image(image, obj)
            if identity_image is not None:
                lookups.append(IdentityLookup(target=target, image=identity_image))

        _save_identity_debug(
            self.config,
            image,
            matched,
            selected,
            source,
            crop.label,
            [lookup.image for lookup in lookups],
            target_crop=crop,
        )
        return lookups

    def _fallback_objects(
        self,
        image: np.ndarray,
        config: IdentityFallbackConfig,
        crop: Crop,
    ) -> tuple[list[DetectedObject], list[DetectedObject], list[DetectedObject]]:
        from ultralytics import YOLO

        with self.fallback_lock:
            if self.fallback_model is None:
                self.fallback_model = YOLO(config.model, task="segment")
                self.logger.info("Loaded identity fallback segment model %s", config.model)

            results = self.fallback_model.predict(
                source=image,
                conf=config.confidence,
                imgsz=config.imgsz,
                stream=False,
                verbose=False,
            )
        if not results:
            return [], [], []

        detected = objects_from_result(results[0], image.shape[:2])
        matched = [
            obj
            for obj in detected
            if obj.mask is not None
            and _label_matches(obj.crop.label, config.labels)
            and (obj.crop.confidence or 0) >= config.confidence
        ]
        centered = sorted(
            [obj for obj in matched if _object_center_in_crop(obj, crop)],
            key=lambda obj: _center_distance_to_crop(obj, crop),
        )
        selected = centered if self.config.multiple else centered[:1]
        return detected, matched, selected


@dataclass
class IdentityLookup:
    target: DetectedObject
    image: np.ndarray


def _label_matches(label: str | None, labels: list[str] | None) -> bool:
    return labels is None or label in set(labels)


def _object_identity_image(image: np.ndarray, obj: DetectedObject) -> np.ndarray | None:
    if obj.mask is not None:
        return _crop_to_mask(apply_mask(image, obj.mask, "gray"), obj.mask)
    return _crop_to_box(image, obj.crop)


def _crop_to_box(image: np.ndarray, crop: Crop) -> np.ndarray | None:
    height, width = image.shape[:2]
    x1 = max(0, min(width, crop.x1))
    x2 = max(0, min(width, crop.x2))
    y1 = max(0, min(height, crop.y1))
    y2 = max(0, min(height, crop.y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return image[y1:y2, x1:x2]


def _crop_to_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None

    x1 = int(xs.min())
    x2 = int(xs.max()) + 1
    y1 = int(ys.min())
    y2 = int(ys.max()) + 1
    if x2 <= x1 or y2 <= y1:
        return None
    return image[y1:y2, x1:x2]


def _center_distance_to_crop(obj: DetectedObject, crop: Crop) -> float:
    target_x = (crop.x1 + crop.x2) / 2
    target_y = (crop.y1 + crop.y2) / 2
    crop_center_x = (obj.crop.x1 + obj.crop.x2) / 2
    crop_center_y = (obj.crop.y1 + obj.crop.y2) / 2
    return (crop_center_x - target_x) ** 2 + (crop_center_y - target_y) ** 2


def _object_center_in_crop(obj: DetectedObject, crop: Crop) -> bool:
    center_x = (obj.crop.x1 + obj.crop.x2) / 2
    center_y = (obj.crop.y1 + obj.crop.y2) / 2
    return crop.x1 <= center_x <= crop.x2 and crop.y1 <= center_y <= crop.y2


def _save_identity_debug(
    config: DetectorIdentityConfig,
    image: np.ndarray,
    objects: list[DetectedObject],
    selected: list[DetectedObject],
    source: str,
    crop_label: str | None,
    identity_images: list[np.ndarray] | None = None,
    target_crop: Crop | None = None,
) -> None:
    if config.debug_directory is None:
        return

    try:
        config.debug_directory.mkdir(parents=True, exist_ok=True)
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
        cv2.imwrite(str(config.debug_directory / filename), overlay)
        for index, identity_image in enumerate(identity_images or [], start=1):
            suffix = (
                "-megadescriptor.png"
                if len(identity_images or []) == 1
                else f"-megadescriptor-{index}.png"
            )
            cv2.imwrite(
                str(config.debug_directory / filename.replace(".jpg", suffix)),
                identity_image,
            )
    except Exception:
        logging.getLogger(__name__).exception("Failed to write identity debug image")


def _object_label(label: str | None, confidence: float | None) -> str:
    if confidence is None:
        return label or "unknown"
    return f"{label or 'unknown'} {confidence:.2f}"


def _format_detected_labels(objects: list[DetectedObject]) -> str:
    if not objects:
        return "[]"

    labels = [_object_label(obj.crop.label, obj.crop.confidence) for obj in objects]
    return "[" + ", ".join(labels) + "]"


def _safe_filename(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_." else "_" for char in value)[
        :80
    ]
