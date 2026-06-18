import logging
from dataclasses import dataclass
from datetime import datetime
from threading import Lock

import cv2
import numpy as np
from aidetector.detection.yolo import YoloObject, apply_mask, objects_from_result
from aidetector.identity.store import SQLiteIdentityStore
from aidetector.media.video import get_crop
from aidetector.utils.config import (
    Detection,
    IdentityProviderConfig,
    IdentityResult,
    max_confidence,
)


@dataclass
class _TrackedIdentityImage:
    frame_index: int
    frame: np.ndarray
    crop_label: str | None
    obj: YoloObject
    identity_image: np.ndarray


class WildlifeToolsIdentityProvider:
    logger = logging.getLogger(__name__)

    def __init__(self, config: IdentityProviderConfig):
        self.config = config
        self.store = SQLiteIdentityStore(
            config.database,
            config.id,
            _model_signature(config),
        )
        self.lock = Lock()
        self.extractor = None
        self.segmenter = None
        self.device = None

    def identify(
        self,
        detection: Detection | list[Detection],
        source: str,
        multiple: bool = False,
    ) -> IdentityResult | list[IdentityResult] | None:
        with self.lock:
            detections = detection if isinstance(detection, list) else [detection]
            if not detections:
                return None

            if len(detections) > 1:
                image_groups = self._event_identity_image_groups(
                    detections,
                    source,
                    multiple,
                )
                if not image_groups:
                    return None

                identities = [
                    self._identify_embedding(
                        _average_embeddings([self._embed(image) for image in images]),
                        source,
                    )
                    for images in image_groups
                    if images
                ]
                if not identities:
                    return None

                self.logger.info(
                    "Identifying from %s sampled identity frame(s) across %s track(s)",
                    sum(len(images) for images in image_groups),
                    len(image_groups),
                )
                return identities if multiple else identities[0]

            identity_images = self._identity_images(
                detections[0],
                source,
                multiple=multiple,
            )
            if not identity_images:
                return None
            identities = [
                self._identify_embedding(self._embed(identity_image), source)
                for identity_image in identity_images
            ]
            return identities if multiple else identities[0]

    def close(self) -> None:
        self.store.close()

    def _embed(self, image: np.ndarray) -> np.ndarray:
        import torch
        import timm
        from wildlife_tools.features import DeepFeatures

        if self.extractor is None:
            self.device = _get_device(torch)
            model = timm.create_model(
                self.config.model,
                num_classes=0,
                pretrained=True,
            )
            self.extractor = DeepFeatures(
                model,
                batch_size=1,
                num_workers=0,
                device=self.device,
            )
            self.logger.info(
                "Loaded identity model %s on %s", self.config.model, self.device
            )

        tensor = _preprocess(image, torch)
        features = self.extractor(_SingleImageDataset(tensor)).features
        return np.asarray(features[0], dtype=np.float32).reshape(-1)

    def _identify_embedding(self, embedding: np.ndarray, source: str) -> IdentityResult:
        return self.store.identify(
            embedding,
            source=source,
            match_threshold=self.config.match_threshold,
            candidate_threshold=self.config.candidate_threshold,
            create_after=self.config.create_after,
        )

    def _identity_images(
        self,
        detection: Detection,
        source: str,
        multiple: bool = False,
    ) -> list[np.ndarray]:
        image = get_crop(
            detection,
            padding=self.config.crop_padding,
            plot=False,
            aspect_ratio=None,
        )
        if image is None:
            self.logger.info("Skipping identity: no detection crop found")
            return []

        if self.config.segment_model is None:
            return [image]

        identity_images = self._mask_identity_images(
            image,
            source,
            detection.images.crop.label if detection.images.crop else None,
            multiple,
        )
        if not identity_images:
            self.logger.info(
                "Skipping identity: no segmentation mask found for labels %s",
                self.config.segment_labels,
            )
        return identity_images

    def _event_identity_image_groups(
        self,
        detections: list[Detection],
        source: str,
        multiple: bool,
    ) -> list[list[np.ndarray]]:
        if self.config.segment_model is not None:
            try:
                tracked_groups = self._tracked_identity_image_groups(
                    detections,
                    source,
                    multiple,
                )
            except Exception:
                self.logger.exception(
                    "Identity tracking failed; falling back to best sampled frame"
                )
                tracked_groups = []
            if tracked_groups:
                return tracked_groups

            self.logger.info(
                "Identity tracking produced no usable track; falling back to best sampled frame"
            )
            best_detection = max(detections, key=lambda item: max_confidence(item.confidence))
            images = self._identity_images(best_detection, source, multiple=False)
            return [images] if images else []

        identity_images = []
        for detection in detections:
            identity_images.extend(self._identity_images(detection, source, multiple=False))
        return [identity_images] if identity_images else []

    def _tracked_identity_image_groups(
        self,
        detections: list[Detection],
        source: str,
        multiple: bool,
    ) -> list[list[np.ndarray]]:
        frames: list[tuple[np.ndarray, str | None]] = []
        for detection in detections:
            image = get_crop(
                detection,
                padding=self.config.crop_padding,
                plot=False,
                aspect_ratio=None,
            )
            if image is not None:
                frames.append((
                    image,
                    detection.images.crop.label if detection.images.crop else None,
                ))

        if not frames:
            self.logger.info("Skipping identity tracking: no detection crops found")
            return []

        results = self._track_segments([frame for frame, _ in frames])
        if not results:
            return []

        labels = set(self.config.segment_labels)
        tracks: dict[int, list[_TrackedIdentityImage]] = {}
        for frame_index, ((frame, crop_label), result) in enumerate(zip(frames, results)):
            debug_objects = objects_from_result(result, frame.shape[:2])
            objects = [
                obj
                for obj in objects_from_result(
                    result,
                    frame.shape[:2],
                    min_confidence=self.config.segment_confidence,
                )
                if (
                    obj.crop.label in labels
                    and obj.mask is not None
                    and obj.track_id is not None
                )
            ]
            if not objects:
                self._save_segment_debug(
                    frame,
                    debug_objects,
                    source,
                    crop_label,
                    selected=False,
                )
                continue

            for obj in objects:
                identity_image = _masked_object_image(
                    frame,
                    obj,
                    self.config.segment_background,
                )
                if identity_image is None or obj.track_id is None:
                    continue

                tracks.setdefault(obj.track_id, []).append(
                    _TrackedIdentityImage(
                        frame_index=frame_index,
                        frame=frame,
                        crop_label=crop_label,
                        obj=obj,
                        identity_image=identity_image,
                    )
                )

        if not tracks:
            self.logger.info(
                "Identity segment tracker found no tracked %s masks",
                self.config.segment_labels,
            )
            return []

        track_groups = sorted(
            tracks.values(),
            key=_track_score,
            reverse=True,
        )
        selected_groups = track_groups if multiple else track_groups[:1]
        for group in selected_groups:
            for sample in group:
                self._save_segment_debug(
                    sample.frame,
                    [sample.obj],
                    source,
                    sample.crop_label,
                    selected=True,
                    identity_image=sample.identity_image,
                )

        return [
            [sample.identity_image for sample in sorted(group, key=lambda item: item.frame_index)]
            for group in selected_groups
        ]

    def _track_segments(self, frames: list[np.ndarray]):
        from ultralytics import YOLO

        if self.segmenter is None:
            self.segmenter = YOLO(self.config.segment_model, task="segment")
            self.logger.info(
                "Loaded identity segment model %s", self.config.segment_model
            )

        return self.segmenter.track(
            source=frames,
            conf=self._segment_prediction_confidence(),
            persist=False,
            stream=False,
            verbose=False,
        )

    def _mask_identity_images(
        self,
        image: np.ndarray,
        source: str,
        crop_label: str | None,
        multiple: bool = False,
    ) -> list[np.ndarray]:
        from ultralytics import YOLO

        if self.segmenter is None:
            self.segmenter = YOLO(self.config.segment_model, task="segment")
            self.logger.info(
                "Loaded identity segment model %s", self.config.segment_model
            )

        results = self.segmenter.predict(
            source=image,
            conf=self._segment_prediction_confidence(),
            stream=False,
            verbose=False,
        )
        if not results:
            return []

        objects = objects_from_result(
            results[0],
            image.shape[:2],
            min_confidence=self.config.segment_confidence,
        )
        debug_objects = objects_from_result(results[0], image.shape[:2])
        labels = set(self.config.segment_labels)
        objects = [
            obj
            for obj in objects
            if obj.crop.label in labels and obj.mask is not None
        ]
        if not objects:
            self.logger.info(
                "Identity segment model %s found labels %s, but none matched %s with masks",
                self.config.segment_model,
                _format_detected_labels(debug_objects),
                self.config.segment_labels,
            )
            self._save_segment_debug(
                image,
                debug_objects,
                source,
                crop_label,
                selected=False,
            )
            return []

        selected_objects = (
            sorted(objects, key=lambda obj: obj.area, reverse=True)
            if multiple
            else [max(objects, key=lambda obj: obj.area)]
        )
        identity_images = []
        for selected in selected_objects:
            identity_image = _masked_object_image(
                image,
                selected,
                self.config.segment_background,
            )
            if identity_image is None:
                continue

            self._save_segment_debug(
                image,
                [selected],
                source,
                crop_label,
                selected=True,
                identity_image=identity_image,
            )
            identity_images.append(identity_image)
        return identity_images

    def _segment_prediction_confidence(self) -> float:
        return (
            min(self.config.segment_confidence, 0.05)
            if self.config.debug_directory is not None
            else self.config.segment_confidence
        )

    def _save_segment_debug(
        self,
        image: np.ndarray,
        objects,
        source: str,
        crop_label: str | None,
        selected: bool,
        identity_image: np.ndarray | None = None,
    ) -> None:
        if self.config.debug_directory is None:
            return

        try:
            self.config.debug_directory.mkdir(parents=True, exist_ok=True)
            raw = image.copy()
            overlay = image.copy()
            color = (0, 180, 255) if selected else (0, 0, 255)
            for obj in objects:
                crop = obj.crop
                if obj.mask is not None:
                    mask_layer = np.zeros_like(overlay)
                    mask_layer[obj.mask] = color
                    overlay = cv2.addWeighted(overlay, 1.0, mask_layer, 0.35, 0)
                cv2.rectangle(overlay, (crop.x1, crop.y1), (crop.x2, crop.y2), color, 2)
                label = crop.label or "unknown"
                if obj.track_id is not None:
                    label = f"{label} #{obj.track_id}"
                if crop.confidence is not None:
                    label = f"{label} {crop.confidence:.2f}"
                cv2.putText(
                    overlay,
                    label,
                    (crop.x1, max(crop.y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                    cv2.LINE_AA,
                )

            caption = (
                f"detector={crop_label or 'unknown'} "
                f"segment_labels={','.join(self.config.segment_labels)}"
            )
            cv2.putText(
                overlay,
                caption,
                (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            combined = np.concatenate([raw, overlay], axis=1)
            filename = (
                f"{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}-"
                f"{_safe_filename(source)}-{_safe_filename(crop_label or 'unknown')}-"
                f"{'matched' if selected else 'failed'}.jpg"
            )
            cv2.imwrite(str(self.config.debug_directory / filename), combined)
            if identity_image is not None:
                cv2.imwrite(
                    str(
                        self.config.debug_directory
                        / filename.replace(".jpg", "-megadescriptor.png")
                    ),
                    identity_image,
                )
        except Exception:
            self.logger.exception("Failed to write identity segment debug image")


class _SingleImageDataset:
    col_label = "identity"

    def __init__(self, image):
        import pandas as pd

        self.image = image
        self.metadata = pd.DataFrame([{"identity": "query"}])

    def __len__(self):
        return 1

    def __getitem__(self, index):
        if index != 0:
            raise IndexError(index)
        return self.image, 0


def _preprocess(image: np.ndarray, torch):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(resized).float().permute(2, 0, 1) / 255
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (tensor - mean) / std


def _average_embeddings(embeddings: list[np.ndarray]) -> np.ndarray:
    normalized = []
    for embedding in embeddings:
        embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(embedding)
        normalized.append(embedding if norm == 0 else embedding / norm)
    return np.mean(normalized, axis=0, dtype=np.float32)


def _masked_object_image(
    image: np.ndarray,
    obj: YoloObject,
    background: str,
) -> np.ndarray | None:
    if obj.mask is None:
        return None

    masked = apply_mask(
        image,
        obj.mask,
        background,
    )
    return _crop_to_mask(masked, obj.mask)


def _track_score(samples: list[_TrackedIdentityImage]) -> tuple[int, float, float]:
    confidences = [
        sample.obj.crop.confidence
        for sample in samples
        if sample.obj.crop.confidence is not None
    ]
    average_confidence = float(np.mean(confidences)) if confidences else 0
    average_area = float(np.mean([sample.obj.area for sample in samples]))
    return len(samples), average_confidence, average_area


def _model_signature(config: IdentityProviderConfig) -> str:
    parts = [f"model={config.model}"]
    if config.segment_model is not None:
        parts.extend(
            [
                f"segment_model={config.segment_model}",
                f"segment_labels={','.join(config.segment_labels)}",
                f"segment_confidence={config.segment_confidence}",
                f"segment_background={config.segment_background}",
                "segment_crop=mask",
            ]
        )
    return "|".join(parts)


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


def _format_detected_labels(objects) -> str:
    if not objects:
        return "[]"

    labels = [
        f"{obj.crop.label}:{obj.crop.confidence:.2f}"
        if obj.crop.confidence is not None
        else str(obj.crop.label)
        for obj in objects
    ]
    return "[" + ", ".join(labels) + "]"


def _safe_filename(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_." else "_" for char in value)[
        :80
    ]


def _get_device(torch):
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
