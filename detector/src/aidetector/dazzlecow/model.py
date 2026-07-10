from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from numpy import ndarray

IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406))[:, None, None]
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225))[:, None, None]


class DazzleCowEncoder:
    def __init__(
        self,
        checkpoint: Path | None = None,
        *,
        device: str = "auto",
        feature_dim: int = 128,
        pretrained: bool = False,
    ):
        self.device = resolve_device(device)
        payload = (
            torch.load(checkpoint, map_location="cpu", weights_only=False)
            if checkpoint is not None
            else None
        )
        if isinstance(payload, dict):
            feature_dim = int(payload.get("feature_dim", feature_dim))
        self.feature_dim = feature_dim
        self.model = _resnet50(feature_dim, pretrained=pretrained)
        if payload is not None:
            self._load(payload, checkpoint)
        self.model.to(self.device).eval()

    def embed(self, images: list[ndarray]) -> ndarray:
        if not images:
            return np.empty((0, self.feature_dim), dtype=np.float32)

        batch = torch.stack([image_tensor(image) for image in images]).to(self.device)
        with torch.inference_mode():
            embeddings = torch.nn.functional.normalize(self.model(batch), dim=1)
        return embeddings.cpu().numpy().astype(np.float32)

    def _load(self, payload: Any, checkpoint: Path | None) -> None:
        state_dict = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
        if not isinstance(state_dict, dict):
            raise ValueError(f"Invalid DazzleCow checkpoint: {checkpoint}")

        state_dict = {
            key.removeprefix("model.").removeprefix("net."): value
            for key, value in state_dict.items()
        }
        self.model.load_state_dict(state_dict)


def create_encoder_model(feature_dim: int = 128, pretrained: bool = True):
    return _resnet50(feature_dim, pretrained=pretrained)


def image_tensor(image: ndarray, size: int = 256) -> torch.Tensor:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(resized.copy()).permute(2, 0, 1).float().div(255)
    return (tensor - IMAGENET_MEAN) / IMAGENET_STD


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    return torch.device(device)


def _resnet50(feature_dim: int, pretrained: bool):
    from torchvision.models import ResNet50_Weights, resnet50

    weights: Any = ResNet50_Weights.DEFAULT if pretrained else None
    model = resnet50(weights=weights)
    model.fc = torch.nn.Sequential(
        model.fc,
        torch.nn.ReLU(),
        torch.nn.Linear(1000, feature_dim),
    )
    return model
