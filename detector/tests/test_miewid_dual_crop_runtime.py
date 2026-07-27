from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from aidetector.reid.miewid import (
    MiewIdDualCropEncoder,
    _load_pytorch_model,
    _PyTorchEmbeddingBackend,
    _select_pytorch_device,
    dual_crop_normalized_mean,
    normalized_prototype,
    preprocess_miewid_rgb,
    white_square_letterbox_rgb,
)
from aidetector.reid.models import (
    MIEWID_EMBEDDING_DIMENSION,
    MODEL_REGISTRY,
    ResolvedModelAsset,
)


class FakeBackend:
    active_device = "cpu"

    def __init__(self):
        self.inputs: list[np.ndarray] = []

    def encode(self, values: np.ndarray) -> np.ndarray:
        self.inputs.append(values.copy())
        result = np.zeros(
            (len(values), MIEWID_EMBEDDING_DIMENSION),
            dtype=np.float32,
        )
        result[: len(values) // 2, 0] = 1.0
        result[len(values) // 2 :, 1] = 1.0
        return result


def fake_asset(tmp_path: Path) -> ResolvedModelAsset:
    specification = MODEL_REGISTRY["miewid-dual-crop-v1"]
    return ResolvedModelAsset(
        specification,
        tmp_path / "model.safetensors",
        tmp_path / "manifest.json",
    )


def test_white_square_letterbox_is_centered_without_source_rescaling():
    rgb = np.zeros((2, 4, 3), dtype=np.uint8)
    rgb[:, :, 0] = 17

    result = white_square_letterbox_rgb(rgb)

    assert result.shape == (4, 4, 3)
    assert np.array_equal(result[1:3], rgb)
    assert np.all(result[0] == 255)
    assert np.all(result[3] == 255)


def test_preprocessing_is_exact_opencv_rgb_inter_linear_imagenet():
    rgb = np.asarray(
        [
            [[0, 64, 255], [255, 128, 0]],
            [[32, 16, 8], [200, 100, 50]],
        ],
        dtype=np.uint8,
    )

    result = preprocess_miewid_rgb(rgb)

    resized = cv2.resize(rgb, (440, 440), interpolation=cv2.INTER_LINEAR)
    expected = resized.astype(np.float32) / 255.0
    expected = (
        expected - np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
    ) / np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
    expected = np.ascontiguousarray(expected.transpose(2, 0, 1))
    assert result.dtype == np.float32
    assert result.shape == (3, 440, 440)
    assert np.array_equal(result, expected)


def test_dual_crop_fusion_is_float64_equal_mean_then_float32():
    canonical = np.zeros((2, MIEWID_EMBEDDING_DIMENSION), dtype=np.float32)
    letterbox = np.zeros_like(canonical)
    canonical[:, 0] = 1.0
    letterbox[:, 1] = 1.0

    result = dual_crop_normalized_mean(canonical, letterbox)

    expected = np.float32(1.0 / np.sqrt(2.0))
    assert result.dtype == np.float32
    assert np.allclose(result[:, 0], expected, atol=1e-7)
    assert np.allclose(result[:, 1], expected, atol=1e-7)


def test_query_and_gallery_prototypes_use_the_frozen_float_path():
    values = np.zeros((4, MIEWID_EMBEDDING_DIMENSION), dtype=np.float32)
    values[0:2, 0] = 1
    values[2:4, 1] = 1

    prototype = normalized_prototype(values)

    mean = np.mean(values.astype(np.float64), axis=0, keepdims=True).astype(np.float32)
    expected = mean / np.linalg.norm(mean, axis=1, keepdims=True)
    assert np.array_equal(prototype, expected[0])


def test_encoder_executes_canonical_and_letterbox_once_per_crop(tmp_path: Path):
    backend = FakeBackend()
    encoder = MiewIdDualCropEncoder(fake_asset(tmp_path), backend=backend)
    crops = [
        np.full((20, 40, 3), (20, 30, 40), dtype=np.uint8),
        np.full((30, 15, 3), (40, 50, 60), dtype=np.uint8),
    ]

    result = encoder.embed(crops)

    assert result.shape == (2, MIEWID_EMBEDDING_DIMENSION)
    assert len(backend.inputs) == 1
    assert backend.inputs[0].shape == (4, 3, 440, 440)
    assert np.allclose(
        np.linalg.norm(result.astype(np.float64), axis=1),
        1.0,
        atol=2e-5,
    )


def test_device_selection_falls_back_to_cpu_off_apple_silicon(monkeypatch):
    monkeypatch.setattr("aidetector.reid.miewid.platform.system", lambda: "Linux")
    monkeypatch.setattr("aidetector.reid.miewid.platform.machine", lambda: "x86_64")

    assert _select_pytorch_device() == "cpu"


def test_encoder_rejects_unreviewed_backend_device(tmp_path: Path):
    backend = FakeBackend()
    backend.active_device = "cuda"

    with pytest.raises(RuntimeError, match="unreviewed device"):
        MiewIdDualCropEncoder(fake_asset(tmp_path), backend=backend)


def test_cpu_backend_uses_reviewed_float32_fallback(tmp_path: Path):
    import torch

    backend = _PyTorchEmbeddingBackend(
        fake_asset(tmp_path),
        device="cpu",
        model=torch.nn.Identity(),
    )

    assert backend.active_device == "cpu"
    assert backend.inference_dtype is torch.float32


def test_incomplete_safetensors_state_dictionary_fails_closed(tmp_path: Path):
    import torch
    from safetensors.torch import save_file

    asset = fake_asset(tmp_path)
    save_file({"unexpected": torch.zeros(1)}, str(asset.checkpoint))

    with pytest.raises(ValueError, match="key count changed"):
        _load_pytorch_model(asset)


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((0, 2, 3), dtype=np.uint8),
        np.zeros((2, 2), dtype=np.uint8),
        np.zeros((2, 2, 3), dtype=np.float32),
    ],
)
def test_preprocessing_rejects_noncanonical_crop_contract(bad: Any):
    with pytest.raises(ValueError):
        preprocess_miewid_rgb(bad)
