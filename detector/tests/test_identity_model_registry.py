import json
from pathlib import Path

import pytest
from aidetector.reid import models as model_registry

from aidetector.reid.models import (
    MIEWID_CONFIG_SOURCE_SHA256,
    MIEWID_DUAL_CROP_V1,
    MIEWID_EMBEDDING_DIMENSION,
    MIEWID_INPUT_SIZE,
    MIEWID_PYTHON_SOURCE_SHA256,
    MIEWID_PYTORCH_REVISION,
    MIEWID_PYTORCH_SHA256,
    MIEWID_PYTORCH_SIZE_BYTES,
    MIEWID_STATE_KEY_COUNT,
    MODEL_REGISTRY,
    ModelDownloadError,
    resolve_model_asset,
)


def test_cow_preset_selects_only_the_reviewed_encoder_key():
    root = Path(__file__).resolve().parents[2]
    preset = json.loads((root / "config/identity/cow.json").read_text())

    assert preset["encoder"] == MIEWID_DUAL_CROP_V1
    assert not any(
        key in preset
        for key in (
            "checkpoint",
            "checksum",
            "model_path",
            "providers",
            "preprocessing",
        )
    )


def test_model_registry_owns_exact_m7zq_execution_provenance():
    asset = MODEL_REGISTRY[MIEWID_DUAL_CROP_V1]

    assert asset.checkpoint == (
        "models/miewid/miewid_msv3_official_4f1d7f2b.safetensors"
    )
    assert asset.manifest == "miewid-dual-crop-v1.json"
    assert asset.sha256 == MIEWID_PYTORCH_SHA256
    assert asset.size_bytes == MIEWID_PYTORCH_SIZE_BYTES
    assert asset.immutable_revision == MIEWID_PYTORCH_REVISION
    assert asset.python_source_sha256 == MIEWID_PYTHON_SOURCE_SHA256
    assert asset.config_source_sha256 == MIEWID_CONFIG_SOURCE_SHA256
    assert asset.state_key_count == MIEWID_STATE_KEY_COUNT
    assert asset.input_size == MIEWID_INPUT_SIZE
    assert asset.embedding_dimension == MIEWID_EMBEDDING_DIMENSION
    assert asset.runtime_backend == "pytorch"
    assert asset.download.repository == "conservationxlabs/miewid-msv3"
    assert asset.download.revision == MIEWID_PYTORCH_REVISION
    assert asset.download.filename == "model.safetensors"
    assert asset.download.url == (
        "https://huggingface.co/conservationxlabs/miewid-msv3/resolve/"
        f"{MIEWID_PYTORCH_REVISION}/model.safetensors"
    )
    assert asset.device_order == ("mps", "cpu")
    assert asset.device_dtypes == (("mps", "float16"), ("cpu", "float32"))
    assert asset.preprocessing.crops == (
        "canonical-resize",
        "white-square-letterbox-resize",
    )
    assert asset.preprocessing.interpolation == "opencv.INTER_LINEAR"
    assert asset.preprocessing.letterbox_value == (255, 255, 255)
    assert asset.preprocessing.fusion_weights == (0.5, 0.5)
    assert asset.preprocessing.normalize_each_embedding
    assert asset.preprocessing.normalize_fused_embedding


def test_unknown_encoder_fails_before_any_runtime_download(tmp_path: Path):
    with pytest.raises(ValueError, match="Unknown identity encoder"):
        resolve_model_asset("network-model", tmp_path)


def test_missing_reviewed_checkpoint_uses_only_pinned_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[Path] = []

    def unavailable_download(_asset, checkpoint: Path) -> None:
        calls.append(checkpoint)
        raise ModelDownloadError("offline")

    monkeypatch.setattr(
        model_registry,
        "_download_checkpoint",
        unavailable_download,
    )

    with pytest.raises(ModelDownloadError, match="offline"):
        resolve_model_asset(MIEWID_DUAL_CROP_V1, tmp_path)

    assert calls == [
        tmp_path / "models/miewid/miewid_msv3_official_4f1d7f2b.safetensors"
    ]


def test_corrupt_reviewed_checkpoint_is_never_replaced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    checkpoint = tmp_path / "models/miewid/miewid_msv3_official_4f1d7f2b.safetensors"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"not-the-reviewed-checkpoint")
    monkeypatch.setattr(
        model_registry,
        "_download_checkpoint",
        lambda *_args: pytest.fail("existing checkpoints must not be downloaded"),
    )

    with pytest.raises(ValueError, match="byte size changed"):
        resolve_model_asset(MIEWID_DUAL_CROP_V1, tmp_path)

    assert checkpoint.read_bytes() == b"not-the-reviewed-checkpoint"
