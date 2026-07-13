from pathlib import Path
from typing import Any

import cv2
import numpy as np
from aidetector.domain.vectors import normalize_rows
from huggingface_hub import hf_hub_download
from numpy import ndarray

MIEWID_REPOSITORY = "james-burgess/miewid"
MIEWID_REVISION = "8ef0f5c426dd089bccc396b7cf07bf9a8fed5140"
MIEWID_FILENAME = "miewid.onnx"
MIEWID_WEIGHTS = "model.safetensors"
MIEWID_MODEL = f"{MIEWID_REPOSITORY}@{MIEWID_REVISION}"
IMAGE_SIZE = 440
FEATURE_DIM = 2152
MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


class OnnxMiewIdEncoder:
    def __init__(self, session: Any | None = None):
        if session is None:
            import onnxruntime as ort

            model = Path(
                hf_hub_download(
                    repo_id=MIEWID_REPOSITORY,
                    filename=MIEWID_FILENAME,
                    revision=MIEWID_REVISION,
                )
            )
            session = ort.InferenceSession(
                str(model),
                providers=ort.get_available_providers(),
            )
        self.session: Any = session
        input_metadata = self.session.get_inputs()[0]
        output_metadata = self.session.get_outputs()[0]
        if list(input_metadata.shape) != [1, 3, IMAGE_SIZE, IMAGE_SIZE]:
            raise ValueError(f"Unexpected MiewID input shape: {input_metadata.shape}")
        if len(output_metadata.shape) != 2 or not isinstance(
            output_metadata.shape[1], int
        ):
            raise ValueError(f"Unexpected MiewID output shape: {output_metadata.shape}")
        self.input_name = input_metadata.name
        self.output_name = output_metadata.name
        self.feature_dim = output_metadata.shape[1]

    def embed(self, images: list[ndarray]) -> ndarray:
        if not images:
            return np.empty((0, self.feature_dim), dtype=np.float32)

        embeddings = [
            np.asarray(
                self.session.run(
                    [self.output_name],
                    {self.input_name: _preprocess(image)},
                )[0][0],
                dtype=np.float32,
            )
            for image in images
        ]
        values = np.asarray(embeddings, dtype=np.float32)
        return normalize_rows(values)


class TorchMiewIdEncoder:
    feature_dim = FEATURE_DIM

    def __init__(self, model: Any | None = None, device: Any | None = None):
        import torch

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model is None:
            model = _load_torch_model()
        self.model = model.to(device).eval()
        self.device = device

    def embed(self, images: list[ndarray]) -> ndarray:
        if not images:
            return np.empty((0, self.feature_dim), dtype=np.float32)

        import torch

        batch = np.concatenate([_preprocess(image) for image in images])
        with torch.inference_mode():
            values = self.model(torch.from_numpy(batch).to(self.device))
        embeddings = values.detach().cpu().numpy().astype(np.float32, copy=False)
        return normalize_rows(embeddings)


def build_miewid_encoder(*, pytorch: bool) -> OnnxMiewIdEncoder | TorchMiewIdEncoder:
    return TorchMiewIdEncoder() if pytorch else OnnxMiewIdEncoder()


def _load_torch_model():
    import timm
    import torch
    from safetensors.torch import load_file
    from torch import nn
    from torch.nn import functional

    class GeM(nn.Module):
        def __init__(self):
            super().__init__()
            self.p = nn.Parameter(torch.full((1,), 3.0))

        def forward(self, values):
            pooled = functional.adaptive_avg_pool2d(
                values.clamp(min=1e-6).pow(self.p),
                1,
            )
            return pooled.pow(1 / self.p)

    class MiewIdNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = timm.create_model(
                "efficientnetv2_rw_m",
                pretrained=False,
                num_classes=0,
            )
            self.backbone.global_pool = GeM()
            self.bn = nn.BatchNorm1d(FEATURE_DIM)
            # Retain the unused training head so the published checkpoint loads strictly.
            self.final = nn.Linear(FEATURE_DIM, 10)

        def forward(self, values):
            features = self.backbone(values).view(values.shape[0], -1)
            return self.bn(features)

    weights = Path(
        hf_hub_download(
            repo_id=MIEWID_REPOSITORY,
            filename=MIEWID_WEIGHTS,
            revision=MIEWID_REVISION,
        )
    )
    model = MiewIdNet()
    model.load_state_dict(load_file(weights), strict=True)
    return model


def _preprocess(image: ndarray) -> ndarray:
    resized = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    normalized = (rgb - MEAN) / STD
    return np.ascontiguousarray(normalized.transpose(2, 0, 1)[None])
