import logging
from threading import Lock

import cv2
import numpy as np
from aidetector.identity.store import SQLiteIdentityStore
from aidetector.utils.config import (
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

    def match(
        self,
        image: np.ndarray,
    ) -> IdentityResult | None:
        with self.lock:
            return self.store.match(
                self._embed(image),
                match_threshold=self.config.match_threshold,
            )

    def identify(
        self,
        image: np.ndarray,
    ) -> IdentityResult | None:
        with self.lock:
            return self._identify_embedding(self._embed(image))

    def update_identity(
        self,
        identity: str,
        image: np.ndarray,
    ) -> IdentityResult | None:
        with self.lock:
            return self.store.update_identity(
                identity,
                self._embed(image),
                match_threshold=self.config.match_threshold,
            )

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

    def _identify_embedding(self, embedding: np.ndarray) -> IdentityResult | None:
        return self.store.identify(
            embedding,
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


def _preprocess(image: np.ndarray, torch):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(resized).float().permute(2, 0, 1) / 255
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (tensor - mean) / std


def _model_signature(config: IdentityProviderConfig) -> str:
    return f"model={config.model}"


def _get_device(torch):
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
