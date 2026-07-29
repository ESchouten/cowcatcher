import numpy as np
import torch

from aidetector.reid.miewid import (
    FEATURE_DIM,
    MODEL_DIRECTORY,
    MODEL_FILENAME,
    MODEL_REPOSITORY,
    MODEL_REVISION,
    MiewIdEncoder,
    _device,
    _download_checkpoint,
)


class FakeModel(torch.nn.Module):
    def forward(self, values):
        result = torch.zeros(
            (len(values), FEATURE_DIM),
            dtype=values.dtype,
            device=values.device,
        )
        middle = len(values) // 2
        result[:middle, 0] = 1
        result[middle:, 1] = 1
        return result


def test_encoder_combines_resized_and_letterboxed_crops():
    encoder = MiewIdEncoder(model=FakeModel(), device="cpu")
    crops = [
        np.full((20, 40, 3), (20, 30, 40), dtype=np.uint8),
        np.full((30, 15, 3), (40, 50, 60), dtype=np.uint8),
    ]

    result = encoder.embed(crops)

    assert result.shape == (2, FEATURE_DIM)
    assert np.allclose(result[:, :2], 1 / np.sqrt(2))
    assert np.allclose(np.linalg.norm(result, axis=1), 1)


def test_device_selection_uses_mps_only_on_apple_silicon(monkeypatch):
    monkeypatch.setattr("aidetector.reid.miewid.platform.system", lambda: "Linux")
    assert _device() == "cpu"

    monkeypatch.setattr("aidetector.reid.miewid.platform.system", lambda: "Darwin")
    monkeypatch.setattr("aidetector.reid.miewid.platform.machine", lambda: "arm64")
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert _device() == "mps"


def test_checkpoint_downloads_into_the_local_models_directory(monkeypatch, tmp_path):
    checkpoint = tmp_path / MODEL_FILENAME
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        return checkpoint

    monkeypatch.setattr("aidetector.reid.miewid.hf_hub_download", fake_download)

    assert _download_checkpoint() == checkpoint
    assert calls == [
        {
            "repo_id": MODEL_REPOSITORY,
            "filename": MODEL_FILENAME,
            "revision": MODEL_REVISION,
            "local_dir": MODEL_DIRECTORY,
        }
    ]
