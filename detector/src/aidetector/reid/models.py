from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

MIEWID_DUAL_CROP_V1 = "miewid-dual-crop-v1"
MIEWID_MODEL_VERSION_ID = "model_miewid_msv3_pytorch_4f1d7f2b_adff92b39678f37eb74861c6"
MIEWID_PYTORCH_REVISION = "4f1d7f2b521149e5fe34bb85f377248ce9971a7d"
MIEWID_PYTHON_SOURCE_SHA256 = (
    "09ef802d44b528acb4848199ef5403dff3b9ac7a7d1c1c4af254f0c84aae5bd7"
)
MIEWID_CONFIG_SOURCE_SHA256 = (
    "be4daa421d2d6781a71ca6ae2ea86732afec823aaeb9166311acf56066cc7464"
)
MIEWID_PYTORCH_SHA256 = (
    "adff92b39678f37eb74861c6399a741639a8907ec2382738e903d6120727b348"
)
MIEWID_PYTORCH_SIZE_BYTES = 205_809_924
MIEWID_STATE_KEY_COUNT = 1_210
MIEWID_ONNX_SHA256 = "43a1252452d15ff030ebf057985e0971f40866d73eacf0af17b08d95e8702764"
MIEWID_ONNX_SIZE_BYTES = 203_973_621
MIEWID_INPUT_SIZE = 440
MIEWID_EMBEDDING_DIMENSION = 2_152


@dataclass(frozen=True, slots=True)
class PreprocessingSpec:
    source_color_space: str
    input_layout: str
    input_dtype: str
    input_size: int
    interpolation: str
    crops: tuple[str, ...]
    letterbox_value: tuple[int, int, int]
    scale: float
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    normalize_each_embedding: bool
    fusion_weights: tuple[float, ...]
    normalize_fused_embedding: bool


@dataclass(frozen=True, slots=True)
class ModelAsset:
    key: str
    checkpoint: str
    manifest: str
    model_version_id: str
    immutable_revision: str
    python_source_sha256: str
    config_source_sha256: str
    sha256: str
    size_bytes: int
    state_key_count: int
    input_size: int
    embedding_dimension: int
    runtime_backend: str
    device_order: tuple[str, ...]
    device_dtypes: tuple[tuple[str, str], ...]
    preprocessing: PreprocessingSpec

    def resolve(self, data_directory: Path) -> ResolvedModelAsset:
        checkpoint = (data_directory / self.checkpoint).resolve()
        manifest = Path(__file__).with_name("model_manifests") / self.manifest
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"Reviewed identity checkpoint is missing: {checkpoint}"
            )
        if checkpoint.stat().st_size != self.size_bytes:
            raise ValueError("Reviewed identity checkpoint byte size changed")
        if _sha256_path(checkpoint) != self.sha256:
            raise ValueError("Reviewed identity checkpoint checksum changed")
        if not manifest.is_file():
            raise FileNotFoundError(
                f"Reviewed identity manifest is missing: {manifest}"
            )
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        architecture_source = payload.get("architecture_source")
        preprocessing = payload.get("preprocessing")
        if (
            payload.get("model_version_id") != self.model_version_id
            or payload.get("immutable_revision") != self.immutable_revision
            or not isinstance(architecture_source, dict)
            or architecture_source.get("modeling_miewid_sha256")
            != self.python_source_sha256
            or architecture_source.get("config_json_sha256")
            != self.config_source_sha256
            or not isinstance(preprocessing, dict)
            or preprocessing.get("inference_dtype_by_device")
            != dict(self.device_dtypes)
            or payload.get("sha256") != self.sha256
            or payload.get("production_enabled") is not False
            or payload.get("private_noncommercial_runtime_enabled") is not True
            or payload.get("operator_supplied_local_asset_only") is not True
            or payload.get("runtime_download_allowed") is not False
            or payload.get("redistribution_allowed") is not False
            or payload.get("licence_status") != "DISPUTED_UNRESOLVED"
        ):
            raise ValueError("Reviewed identity model manifest changed")
        return ResolvedModelAsset(self, checkpoint, manifest)


@dataclass(frozen=True, slots=True)
class ResolvedModelAsset:
    specification: ModelAsset
    checkpoint: Path
    manifest: Path


MODEL_REGISTRY: dict[str, ModelAsset] = {
    MIEWID_DUAL_CROP_V1: ModelAsset(
        key=MIEWID_DUAL_CROP_V1,
        checkpoint="models/miewid/miewid_msv3_official_4f1d7f2b.safetensors",
        manifest="miewid-dual-crop-v1.json",
        model_version_id=MIEWID_MODEL_VERSION_ID,
        immutable_revision=MIEWID_PYTORCH_REVISION,
        python_source_sha256=MIEWID_PYTHON_SOURCE_SHA256,
        config_source_sha256=MIEWID_CONFIG_SOURCE_SHA256,
        sha256=MIEWID_PYTORCH_SHA256,
        size_bytes=MIEWID_PYTORCH_SIZE_BYTES,
        state_key_count=MIEWID_STATE_KEY_COUNT,
        input_size=MIEWID_INPUT_SIZE,
        embedding_dimension=MIEWID_EMBEDDING_DIMENSION,
        runtime_backend="pytorch",
        device_order=("mps", "cpu"),
        device_dtypes=(("mps", "float16"), ("cpu", "float32")),
        preprocessing=PreprocessingSpec(
            source_color_space="RGB",
            input_layout="NCHW",
            input_dtype="float32",
            input_size=MIEWID_INPUT_SIZE,
            interpolation="opencv.INTER_LINEAR",
            crops=("canonical-resize", "white-square-letterbox-resize"),
            letterbox_value=(255, 255, 255),
            scale=1.0 / 255.0,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            normalize_each_embedding=True,
            fusion_weights=(0.5, 0.5),
            normalize_fused_embedding=True,
        ),
    )
}


def resolve_model_asset(key: str, data_directory: Path) -> ResolvedModelAsset:
    try:
        asset = MODEL_REGISTRY[key]
    except KeyError as error:
        raise ValueError(f"Unknown identity encoder: {key}") from error
    return asset.resolve(data_directory)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
