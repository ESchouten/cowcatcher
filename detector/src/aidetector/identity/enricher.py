import logging
from collections.abc import Iterable
from dataclasses import dataclass
from threading import Lock
from time import monotonic

import numpy as np
from aidetector.detection.yolo import apply_mask, objects_from_result
from aidetector.identity.debug import format_detected_labels, save_identity_debug
from aidetector.identity.provider import IdentityProvider
from aidetector.utils.config import (
    Crop,
    DetectedObject,
    Detection,
    DetectorIdentityConfig,
    IdentityFallbackConfig,
    IdentityResult,
)


class IdentityEnricher:
    logger = logging.getLogger(__name__)

    def __init__(self, provider: IdentityProvider, config: DetectorIdentityConfig):
        self.provider = provider
        self.config = config
        self.labels = set(config.labels) if config.labels is not None else None
        self.fallback_model = None
        self.fallback_lock = Lock()
        self.identity_lock = Lock()
        self.identities_by_track: dict[tuple[str, int], IdentityResult] = {}
        self.lookup_attempts: dict[tuple[str, int], float] = {}
        self.update_attempts: dict[tuple[str, int], float] = {}

    def enrich(
        self,
        source: str,
        detection: Detection,
        detections: list[Detection] | None = None,
    ) -> None:
        with self.identity_lock:
            self._enrich(
                source,
                detection,
                detections or [detection],
            )

    def enrich_live(self, source: str, detection: Detection) -> None:
        with self.identity_lock:
            self._enrich(
                source,
                detection,
                [detection],
                lookup_time=monotonic(),
                log_empty=False,
            )

    def _enrich(
        self,
        source: str,
        detection: Detection,
        detections: list[Detection],
        lookup_time: float | None = None,
        log_empty: bool = True,
    ) -> None:
        self._reset_identities(detections)
        self._apply_cached_identities(source, detections)

        if lookup_time is not None:
            self._update_cached_identities(detection, source, lookup_time)

        lookups = self._identity_lookups(
            detection,
            source,
            lookup_time=lookup_time,
        )
        if not lookups:
            if log_empty:
                self.logger.info("Identity provider received no lookup images")
            self._set_detection_identities(detections)
            return

        self.logger.info(
            "Looking up %d identity image(s) with provider %s",
            len(lookups),
            self.config.provider,
        )
        identified_lookups: list[IdentityLookup] = []
        identities: list[IdentityResult] = []
        for lookup in lookups:
            identity = self.provider.identify(lookup.image)
            if identity is None:
                continue
            identified_lookups.append(lookup)
            identities.append(identity)
        if not identities:
            if log_empty:
                self.logger.info("Identity provider returned no result")
            self._set_detection_identities(detections)
            return

        self._apply_lookup_results(source, detections, identified_lookups, identities)

        primary = identities[0]
        self.logger.info(
            "Identity result: status=%s id=%s similarity=%s",
            primary.status,
            primary.identity,
            primary.similarity,
        )
        self._set_detection_identities(detections)
        detection.identities = _unique_identities([*detection.identities, *identities])

    def _reset_identities(self, detections: list[Detection]) -> None:
        for detection in detections:
            detection.identities = []
            for obj in detection.images.objects:
                obj.identity = None

    def _apply_cached_identities(
        self,
        source: str,
        detections: list[Detection],
    ) -> None:
        for detection in detections:
            for obj in detection.images.objects:
                if obj.track_id is None:
                    continue
                obj.identity = self.identities_by_track.get((source, obj.track_id))

    def _set_detection_identities(self, detections: list[Detection]) -> None:
        for detection in detections:
            detection.identities = _unique_identities(
                obj.identity for obj in detection.images.objects
            )

    def _apply_lookup_results(
        self,
        source: str,
        detections: list[Detection],
        lookups: list["IdentityLookup"],
        identities: list[IdentityResult],
    ) -> None:
        identities_by_track_id: dict[int, IdentityResult] = {}
        for lookup, identity in zip(lookups, identities):
            if lookup.track_id is None:
                if lookup.target.identity is None:
                    lookup.target.identity = identity
                continue

            self.identities_by_track[(source, lookup.track_id)] = identity
            identities_by_track_id[lookup.track_id] = identity

        for detection in detections:
            for obj in detection.images.objects:
                if obj.track_id in identities_by_track_id:
                    obj.identity = identities_by_track_id[obj.track_id]

    def _update_cached_identities(
        self,
        detection: Detection,
        source: str,
        lookup_time: float,
    ) -> None:
        updates = self._identity_updates(detection, source, lookup_time)
        if not updates:
            return

        self.logger.info(
            "Updating %d cached identity sample(s) with provider %s",
            len(updates),
            self.config.provider,
        )
        for update in updates:
            identity = self.provider.update_identity(update.identity, update.image)
            if identity is None:
                continue
            update.target.identity = identity
            self.identities_by_track[(source, update.track_id)] = identity

    def _identity_updates(
        self,
        detection: Detection,
        source: str,
        lookup_time: float,
    ) -> list["IdentityUpdate"]:
        objects = [
            obj
            for obj in detection.images.objects
            if obj.track_id is not None
            and obj.mask is not None
            and obj.identity is not None
            and _label_matches(obj.crop.label, self.labels)
        ]
        updates: list[IdentityUpdate] = []
        for obj in _select_objects(objects, self.config.multiple):
            if not self._update_due(source, obj.track_id, lookup_time):
                continue
            image = _object_identity_image(detection.images.jpg, obj)
            if image is not None and obj.identity:
                updates.append(
                    IdentityUpdate(
                        target=obj,
                        image=image,
                        track_id=obj.track_id,
                        identity=obj.identity.identity,
                    )
                )
        return updates

    def _identity_lookups(
        self,
        detection: Detection,
        source: str,
        lookup_time: float | None = None,
    ) -> list["IdentityLookup"]:
        objects = [
            obj
            for obj in detection.images.objects
            if _label_matches(obj.crop.label, self.labels)
        ]
        if not objects:
            self.logger.info("Skipping identity: no detector objects matched")
            return []

        selected = _select_objects(objects, self.config.multiple)
        lookup_objects = [obj for obj in selected if _needs_lookup(obj)]
        if lookup_time is not None:
            lookup_objects = [
                obj
                for obj in lookup_objects
                if obj.track_id is not None
                and self._lookup_due(source, obj.track_id, lookup_time)
            ]
        if not lookup_objects:
            return []

        masked_objects = [obj for obj in lookup_objects if obj.mask is not None]
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
                lookup_objects[0],
            )

        return self._object_lookups(
            detection.images.jpg,
            lookup_objects,
            source,
            debug_label="detector",
        )

    def _lookup_due(self, source: str, track_id: int, now: float) -> bool:
        key = (source, track_id)
        last_attempt = self.lookup_attempts.get(key)
        if (
            last_attempt is not None
            and now - last_attempt < self.config.lookup_interval
        ):
            return False
        self.lookup_attempts[key] = now
        return True

    def _update_due(self, source: str, track_id: int, now: float) -> bool:
        key = (source, track_id)
        last_attempt = self.update_attempts.get(key, self.lookup_attempts.get(key))
        if (
            last_attempt is not None
            and now - last_attempt < self.config.update_interval
        ):
            return False
        self.update_attempts[key] = now
        return True

    def _object_lookups(
        self,
        image: np.ndarray,
        objects: list[DetectedObject],
        source: str,
        debug_label: str | None,
    ) -> list["IdentityLookup"]:
        lookups = [
            IdentityLookup(target=obj, image=identity_image, track_id=obj.track_id)
            for obj in objects
            if (identity_image := _object_identity_image(image, obj)) is not None
        ]
        save_identity_debug(
            self.config,
            image,
            objects,
            objects,
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
        if not selected:
            objects = matched or detected
            if matched:
                self.logger.info(
                    "Skipping identity fallback: %s found %s, but none were centered in %s crop",
                    config.model,
                    format_detected_labels(matched),
                    crop.label,
                )
            else:
                self.logger.info(
                    "Identity fallback model %s found labels %s, but none matched %s with masks",
                    config.model,
                    format_detected_labels(detected),
                    config.labels,
                )
            save_identity_debug(
                self.config,
                image,
                objects,
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
                lookups.append(
                    IdentityLookup(
                        target=target,
                        image=identity_image,
                        track_id=target.track_id,
                    )
                )

        save_identity_debug(
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
        labels = set(config.labels)
        matched = [
            obj
            for obj in detected
            if obj.mask is not None
            and _label_matches(obj.crop.label, labels)
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
    track_id: int | None = None


@dataclass
class IdentityUpdate:
    target: DetectedObject
    image: np.ndarray
    track_id: int
    identity: str


def _label_matches(label: str | None, labels: set[str] | None) -> bool:
    return labels is None or label in labels


def _select_objects(
    objects: list[DetectedObject],
    multiple: bool,
) -> list[DetectedObject]:
    return objects if multiple else objects[:1]


def _needs_lookup(obj: DetectedObject) -> bool:
    return obj.track_id is None or obj.identity is None


def _unique_identities(identities: Iterable[IdentityResult | None]) -> list[IdentityResult]:
    unique: list[IdentityResult] = []
    seen: set[str] = set()
    for identity in identities:
        if identity is None:
            continue
        key = identity.identity
        if key in seen:
            continue
        seen.add(key)
        unique.append(identity)
    return unique


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
