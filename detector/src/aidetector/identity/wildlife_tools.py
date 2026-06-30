import logging
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
)


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
        self.device = None
        self.segmenter = None

    def identify(
        self,
        detection: Detection,
        source: str,
        multiple: bool = False,
    ) -> list[IdentityResult]:
        with self.lock:
            identity_images = self._identity_images(detection, source, multiple)
            return [
                self._identify_embedding(self._embed(image), source)
                for image in identity_images
            ]

    def close(self) -> None:
        self.store.close()

    def _identity_images(
        self,
        detection: Detection,
        source: str,
        multiple: bool,
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

        return self._segmented_identity_images(
            image,
            source,
            detection.images.best_crop.label if detection.images.best_crop else None,
            multiple,
        )

    def _segmented_identity_images(
        self,
        image: np.ndarray,
        source: str,
        crop_label: str | None,
        multiple: bool,
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
            imgsz=self.config.segment_imgsz,
            stream=False,
            verbose=False,
        )
        if not results:
            return []

        debug_objects = objects_from_result(results[0], image.shape[:2])
        objects = [
            obj
            for obj in objects_from_result(
                results[0],
                image.shape[:2],
                min_confidence=self.config.segment_confidence,
            )
            if obj.crop.label in set(self.config.segment_labels) and obj.mask is not None
        ]
        if not objects:
            self.logger.info(
                "Identity segment model %s found labels %s, but none matched %s with masks",
                self.config.segment_model,
                _format_detected_labels(debug_objects),
                self.config.segment_labels,
            )
            _save_segment_debug(self.config, image, debug_objects, [], source, crop_label)
            return []

        selected = (
            sorted(objects, key=lambda obj: _segment_center_distance(obj, image.shape[:2]))
            if multiple
            else [min(objects, key=lambda obj: _segment_center_distance(obj, image.shape[:2]))]
        )
        identity_images = [
            identity_image
            for obj in selected
            if (
                identity_image := _masked_object_image(
                    image,
                    obj,
                )
            )
            is not None
        ]
        _save_segment_debug(
            self.config,
            image,
            objects,
            selected,
            source,
            crop_label,
            identity_images,
        )
        return identity_images

    def _segment_prediction_confidence(self) -> float:
        return (
            min(self.config.segment_confidence, 0.05)
            if self.config.debug_directory is not None
            else self.config.segment_confidence
        )

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


def _save_segment_debug(
    config: IdentityProviderConfig,
    image: np.ndarray,
    objects: list[YoloObject],
    selected: list[YoloObject],
    source: str,
    crop_label: str | None,
    identity_images: list[np.ndarray] | None = None,
) -> None:
    if config.debug_directory is None:
        return

    try:
        config.debug_directory.mkdir(parents=True, exist_ok=True)
        overlay = image.copy()
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
                _segment_label(crop.label, crop.confidence),
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
            f"{'matched' if selected else 'failed'}.jpg"
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


def _segment_label(label: str | None, confidence: float | None) -> str:
    if confidence is None:
        return label or "unknown"
    return f"{label or 'unknown'} {confidence:.2f}"


def _preprocess(image: np.ndarray, torch):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(resized).float().permute(2, 0, 1) / 255
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (tensor - mean) / std


def _masked_object_image(image: np.ndarray, obj: YoloObject) -> np.ndarray | None:
    if obj.mask is None:
        return None

    masked = apply_mask(
        image,
        obj.mask,
        "gray",
    )
    return _crop_to_mask(masked, obj.mask)


def _segment_center_distance(obj: YoloObject, image_shape: tuple[int, int]) -> float:
    height, width = image_shape
    image_center_x = width / 2
    image_center_y = height / 2
    crop_center_x = (obj.crop.x1 + obj.crop.x2) / 2
    crop_center_y = (obj.crop.y1 + obj.crop.y2) / 2
    return (crop_center_x - image_center_x) ** 2 + (
        crop_center_y - image_center_y
    ) ** 2


def _model_signature(config: IdentityProviderConfig) -> str:
    parts = [f"model={config.model}"]
    if config.segment_model is not None:
        parts.extend(
            [
                f"segment_model={config.segment_model}",
                f"segment_labels={','.join(config.segment_labels)}",
                f"segment_confidence={config.segment_confidence}",
                f"segment_imgsz={config.segment_imgsz}",
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
        _segment_label(obj.crop.label, obj.crop.confidence)
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
