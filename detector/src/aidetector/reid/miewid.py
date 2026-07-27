from __future__ import annotations

import logging
import platform
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
from numpy import ndarray

from aidetector.reid.models import (
    MIEWID_DUAL_CROP_V1,
    MIEWID_EMBEDDING_DIMENSION,
    MIEWID_INPUT_SIZE,
    ResolvedModelAsset,
    resolve_model_asset,
)

logger = logging.getLogger(__name__)
MPS_DEVICE = "mps"
CPU_DEVICE = "cpu"


class MiewIdEmbeddingBackend(Protocol):
    active_device: str

    def encode(self, prepared: ndarray) -> ndarray: ...


class MiewIdDualCropEncoder:
    """PyTorch inference implementation of the frozen M7ZQ encoder."""

    feature_dim = MIEWID_EMBEDDING_DIMENSION

    def __init__(
        self,
        asset: ResolvedModelAsset,
        *,
        backend: MiewIdEmbeddingBackend | None = None,
    ):
        self.asset = asset
        if asset.specification.runtime_backend != "pytorch":
            raise ValueError("MiewID runtime backend must be PyTorch")
        self.backend = backend or _PyTorchEmbeddingBackend(asset)
        if self.backend.active_device not in asset.specification.device_order:
            raise RuntimeError("MiewID backend selected an unreviewed device")
        self.active_device = self.backend.active_device

    def embed(self, rgb_crops: list[ndarray]) -> ndarray:
        if not rgb_crops:
            return np.empty((0, self.feature_dim), dtype=np.float32)
        count = len(rgb_crops)
        canonical_inputs = [preprocess_miewid_rgb(crop) for crop in rgb_crops]
        letterbox_inputs = [
            preprocess_miewid_rgb(white_square_letterbox_rgb(crop))
            for crop in rgb_crops
        ]
        prepared = np.ascontiguousarray(
            np.stack(canonical_inputs + letterbox_inputs),
            dtype=np.float32,
        )
        embeddings = self.backend.encode(prepared)
        if embeddings.shape != (count * 2, self.feature_dim):
            raise ValueError("MiewID backend returned the wrong embedding shape")
        canonical = embeddings[:count]
        letterbox = embeddings[count:]
        return dual_crop_normalized_mean(canonical, letterbox)


class _PyTorchEmbeddingBackend:
    def __init__(
        self,
        asset: ResolvedModelAsset,
        *,
        device: str | None = None,
        model: Any | None = None,
    ):
        import torch

        self.active_device = device or _select_pytorch_device()
        if self.active_device not in asset.specification.device_order:
            raise RuntimeError("MiewID PyTorch device is not in the reviewed order")
        dtype_name = dict(asset.specification.device_dtypes).get(self.active_device)
        if dtype_name == "float16":
            self.inference_dtype = torch.float16
        elif dtype_name == "float32":
            self.inference_dtype = torch.float32
        else:
            raise RuntimeError("MiewID inference dtype changed")
        self.model = model or _load_pytorch_model(asset)
        self.model.eval()
        self.model.to(device=self.active_device, dtype=self.inference_dtype)
        logger.info(
            "Using native PyTorch %s %s for MiewID model %s",
            self.active_device.upper(),
            dtype_name.upper(),
            asset.specification.model_version_id,
        )

    def encode(self, prepared: ndarray) -> ndarray:
        import torch

        values = np.asarray(prepared)
        if (
            values.dtype != np.float32
            or values.ndim != 4
            or tuple(values.shape[1:]) != (3, MIEWID_INPUT_SIZE, MIEWID_INPUT_SIZE)
            or not np.all(np.isfinite(values))
        ):
            raise ValueError("Unexpected MiewID PyTorch input contract")
        if not len(values):
            return np.empty((0, MIEWID_EMBEDDING_DIMENSION), dtype=np.float32)
        tensor = torch.from_numpy(np.ascontiguousarray(values)).to(
            device=self.active_device,
            dtype=self.inference_dtype,
        )
        with torch.inference_mode():
            raw = self.model(tensor)
        result = np.asarray(raw.detach().float().cpu().numpy(), dtype=np.float32)
        if result.shape != (len(values), MIEWID_EMBEDDING_DIMENSION):
            raise ValueError(f"Unexpected MiewID output shape: {result.shape}")
        return normalize_embedding_rows(result)


def build_miewid_pytorch_model() -> Any:
    """Build the reviewed inference projection of MiewID-msv3 locally."""

    import torch
    import torch.nn as nn
    import torch.nn.functional as functional
    import timm

    class GeM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.p = nn.Parameter(torch.ones(1) * 3.0)
            self.eps = 1e-6

        def forward(self, values: Any) -> Any:
            powered = values.clamp(min=self.eps).pow(self.p)
            pooled = functional.avg_pool2d(
                powered,
                (values.size(-2), values.size(-1)),
            )
            return pooled.pow(1.0 / self.p)

    class MiewIdModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = timm.create_model(
                "efficientnetv2_rw_m",
                pretrained=False,
                num_classes=0,
            )
            self.backbone.global_pool = GeM()
            self.bn = nn.BatchNorm1d(MIEWID_EMBEDDING_DIMENSION)
            self.final = nn.Linear(MIEWID_EMBEDDING_DIMENSION, 10)

        def forward(self, values: Any) -> Any:
            features = self.backbone(values).view(values.shape[0], -1)
            return self.bn(features)

    return MiewIdModel()


def _load_pytorch_model(asset: ResolvedModelAsset) -> Any:
    from safetensors.torch import load_file

    state = load_file(str(asset.checkpoint), device="cpu")
    if len(state) != asset.specification.state_key_count:
        raise ValueError("Reviewed MiewID state-dictionary key count changed")
    model = build_miewid_pytorch_model()
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise ValueError("Reviewed MiewID state-dictionary strict load failed")
    return model


def _select_pytorch_device() -> str:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return CPU_DEVICE

    import torch

    if torch.backends.mps.is_built() and torch.backends.mps.is_available():
        return MPS_DEVICE
    return CPU_DEVICE


def build_miewid_encoder(
    *,
    model_key: str = MIEWID_DUAL_CROP_V1,
    asset_root: Path,
) -> MiewIdDualCropEncoder:
    return MiewIdDualCropEncoder(resolve_model_asset(model_key, asset_root))


def preprocess_miewid_rgb(rgb: ndarray) -> ndarray:
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("MiewID crops must be RGB uint8 HWC arrays")
    if rgb.shape[0] < 1 or rgb.shape[1] < 1:
        raise ValueError("MiewID crop is empty")
    specification = _model_specification()
    preprocessing = specification.preprocessing
    resized = cv2.resize(
        rgb,
        (preprocessing.input_size, preprocessing.input_size),
        interpolation=cv2.INTER_LINEAR,
    )
    values = resized.astype(np.float32) / 255.0
    mean = np.asarray(preprocessing.mean, dtype=np.float32)
    std = np.asarray(preprocessing.std, dtype=np.float32)
    normalized = (values - mean) / std
    return np.ascontiguousarray(
        normalized.transpose(2, 0, 1),
        dtype=np.float32,
    )


def white_square_letterbox_rgb(rgb: ndarray) -> ndarray:
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("M7ZQ letterbox input must be RGB uint8 HWC")
    height, width = rgb.shape[:2]
    if height < 1 or width < 1:
        raise ValueError("M7ZQ letterbox input is empty")
    preprocessing = _model_specification().preprocessing
    side = max(height, width)
    canvas = np.full(
        (side, side, 3),
        preprocessing.letterbox_value,
        dtype=np.uint8,
    )
    y_offset = (side - height) // 2
    x_offset = (side - width) // 2
    canvas[
        y_offset : y_offset + height,
        x_offset : x_offset + width,
    ] = rgb
    return np.ascontiguousarray(canvas)


def normalize_embedding_rows(values: ndarray) -> ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != MIEWID_EMBEDDING_DIMENSION:
        raise ValueError("MiewID embedding matrix changed shape")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("MiewID embeddings contain non-finite values")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= np.finfo(np.float32).eps):
        raise ValueError("MiewID returned a zero-length embedding")
    result = np.ascontiguousarray(matrix / norms, dtype=np.float32)
    _validate_normalized_embeddings(result)
    return result


def dual_crop_normalized_mean(
    canonical: ndarray,
    letterbox: ndarray,
) -> ndarray:
    if (
        canonical.shape != letterbox.shape
        or canonical.ndim != 2
        or canonical.shape[1] != MIEWID_EMBEDDING_DIMENSION
    ):
        raise ValueError("M7ZQ dual-crop embedding matrices changed shape")
    weights = _model_specification().preprocessing.fusion_weights
    total = weights[0] * canonical.astype(np.float64) + weights[1] * letterbox.astype(
        np.float64
    )
    norms = np.linalg.norm(total, axis=1, keepdims=True)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0.0):
        raise ValueError("M7ZQ cannot normalize a dual-crop embedding")
    result = np.ascontiguousarray(total / norms, dtype=np.float32)
    _validate_normalized_embeddings(result)
    return result


def normalized_prototype(embeddings: ndarray) -> ndarray:
    values = np.asarray(embeddings, dtype=np.float32)
    if (
        values.ndim != 2
        or not len(values)
        or values.shape[1] != MIEWID_EMBEDDING_DIMENSION
    ):
        raise ValueError("M7ZQ prototype requires a non-empty embedding matrix")
    _validate_normalized_embeddings(values)
    mean = np.mean(values.astype(np.float64), axis=0, keepdims=True).astype(np.float32)
    return normalize_embedding_rows(mean)[0]


def _validate_normalized_embeddings(values: ndarray) -> None:
    if (
        values.dtype != np.float32
        or values.ndim != 2
        or values.shape[1] != MIEWID_EMBEDDING_DIMENSION
        or not np.all(np.isfinite(values))
    ):
        raise ValueError("M7ZQ embedding matrix contract changed")
    norms = np.linalg.norm(values.astype(np.float64), axis=1)
    if not np.allclose(norms, 1.0, atol=2e-5, rtol=0.0):
        raise ValueError("M7ZQ embeddings are not L2 normalized")


def _model_specification() -> Any:
    from aidetector.reid.models import MODEL_REGISTRY

    return MODEL_REGISTRY[MIEWID_DUAL_CROP_V1]
