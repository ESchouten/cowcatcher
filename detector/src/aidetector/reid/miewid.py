from __future__ import annotations

import platform
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from aidetector.domain.vectors import normalize_rows
from huggingface_hub import hf_hub_download
from numpy import ndarray

MODEL_KEY = "miewid-dual-crop-v1"
MODEL_REPOSITORY = "conservationxlabs/miewid-msv3"
MODEL_REVISION = "4f1d7f2b521149e5fe34bb85f377248ce9971a7d"
MODEL_FILENAME = "model.safetensors"
MODEL_DIRECTORY = Path("models/miewid")
MODEL_SIGNATURE = f"{MODEL_KEY}:{MODEL_REPOSITORY}@{MODEL_REVISION}:dual-crop"
IMAGE_SIZE = 440
FEATURE_DIM = 2_152
MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)


class MiewIdEncoder:
    feature_dim = FEATURE_DIM

    def __init__(self, model: Any | None = None, device: str | None = None):
        import torch

        self.device = device or _device()
        self.dtype = torch.float16 if self.device == "mps" else torch.float32
        self.model = (model or _load_model()).to(
            device=self.device,
            dtype=self.dtype,
        )
        self.model.eval()

    def embed(self, rgb_crops: list[ndarray]) -> ndarray:
        if not rgb_crops:
            return np.empty((0, FEATURE_DIM), dtype=np.float32)

        import torch

        prepared = np.stack(
            [
                *(_preprocess(crop) for crop in rgb_crops),
                *(_preprocess(_letterbox(crop)) for crop in rgb_crops),
            ]
        )
        tensor = torch.from_numpy(prepared).to(
            device=self.device,
            dtype=self.dtype,
        )
        with torch.inference_mode():
            embeddings = self.model(tensor).float().cpu().numpy()

        count = len(rgb_crops)
        canonical = normalize_rows(embeddings[:count])
        letterboxed = normalize_rows(embeddings[count:])
        fused = canonical.astype(np.float64) + letterboxed.astype(np.float64)
        fused /= np.linalg.norm(fused, axis=1, keepdims=True)
        return fused.astype(np.float32)


def _load_model() -> Any:
    import timm
    import torch
    from safetensors.torch import load_file
    from torch import nn
    from torch.nn import functional

    class GeM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.p = nn.Parameter(torch.full((1,), 3.0))

        def forward(self, values: Any) -> Any:
            return functional.avg_pool2d(
                values.clamp(min=1e-6).pow(self.p),
                values.shape[-2:],
            ).pow(1 / self.p)

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = timm.create_model(
                "efficientnetv2_rw_m",
                pretrained=False,
                num_classes=0,
            )
            self.backbone.global_pool = GeM()
            self.bn = nn.BatchNorm1d(FEATURE_DIM)
            self.final = nn.Linear(FEATURE_DIM, 10)

        def forward(self, values: Any) -> Any:
            return self.bn(self.backbone(values).flatten(1))

    checkpoint = _download_checkpoint()
    model = Model()
    model.load_state_dict(load_file(checkpoint), strict=True)
    return model


def _download_checkpoint() -> Path:
    return Path(
        hf_hub_download(
            repo_id=MODEL_REPOSITORY,
            filename=MODEL_FILENAME,
            revision=MODEL_REVISION,
            local_dir=MODEL_DIRECTORY,
        )
    )


def _device() -> str:
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        import torch

        if torch.backends.mps.is_available():
            return "mps"
    return "cpu"


def _preprocess(rgb: ndarray) -> ndarray:
    resized = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE))
    values = resized.astype(np.float32) / 255
    return np.ascontiguousarray(((values - MEAN) / STD).transpose(2, 0, 1))


def _letterbox(rgb: ndarray) -> ndarray:
    height, width = rgb.shape[:2]
    side = max(height, width)
    canvas = np.full((side, side, 3), 255, dtype=np.uint8)
    y = (side - height) // 2
    x = (side - width) // 2
    canvas[y : y + height, x : x + width] = rgb
    return canvas
