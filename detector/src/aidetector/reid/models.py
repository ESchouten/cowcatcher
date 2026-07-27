from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import requests

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
MODEL_DOWNLOAD_CHUNK_BYTES = 1024 * 1024
MODEL_DOWNLOAD_TIMEOUT_SECONDS = (10.0, 120.0)
MODEL_DOWNLOAD_USER_AGENT = "ai-detector/identity-model-installer"
HUGGING_FACE_ORIGIN = "https://huggingface.co"

logger = logging.getLogger(__name__)


class ModelDownloadError(RuntimeError):
    """Raised when a pinned identity model cannot be installed safely."""


@dataclass(frozen=True, slots=True)
class ModelDownload:
    repository: str
    revision: str
    filename: str

    @property
    def url(self) -> str:
        return (
            f"{HUGGING_FACE_ORIGIN}/{self.repository}/resolve/"
            f"{self.revision}/{self.filename}"
        )

    @property
    def reference(self) -> str:
        return f"{self.repository}@{self.revision}/{self.filename}"


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
    download: ModelDownload
    device_order: tuple[str, ...]
    device_dtypes: tuple[tuple[str, str], ...]
    preprocessing: PreprocessingSpec

    def resolve(self, data_directory: Path) -> ResolvedModelAsset:
        checkpoint = (data_directory / self.checkpoint).resolve()
        manifest = Path(__file__).with_name("model_manifests") / self.manifest
        self._validate_manifest(manifest)
        if not checkpoint.exists():
            _download_checkpoint(self, checkpoint)
        if not checkpoint.is_file():
            raise ValueError("Reviewed identity checkpoint path is not a file")
        if checkpoint.stat().st_size != self.size_bytes:
            raise ValueError("Reviewed identity checkpoint byte size changed")
        if _sha256_path(checkpoint) != self.sha256:
            raise ValueError("Reviewed identity checkpoint checksum changed")
        return ResolvedModelAsset(self, checkpoint, manifest)

    def _validate_manifest(self, manifest: Path) -> None:
        if not manifest.is_file():
            raise FileNotFoundError(
                f"Reviewed identity manifest is missing: {manifest}"
            )
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        architecture_source = payload.get("architecture_source")
        download = payload.get("download")
        preprocessing = payload.get("preprocessing")
        if (
            payload.get("model_version_id") != self.model_version_id
            or payload.get("immutable_revision") != self.immutable_revision
            or payload.get("size_bytes") != self.size_bytes
            or not isinstance(architecture_source, dict)
            or architecture_source.get("modeling_miewid_sha256")
            != self.python_source_sha256
            or architecture_source.get("config_json_sha256")
            != self.config_source_sha256
            or not isinstance(download, dict)
            or download.get("provider") != "huggingface"
            or download.get("repository") != self.download.repository
            or download.get("revision") != self.download.revision
            or download.get("filename") != self.download.filename
            or download.get("url") != self.download.url
            or not isinstance(preprocessing, dict)
            or preprocessing.get("inference_dtype_by_device")
            != dict(self.device_dtypes)
            or payload.get("sha256") != self.sha256
            or payload.get("production_enabled") is not False
            or payload.get("private_noncommercial_runtime_enabled") is not True
            or payload.get("official_upstream_download_only") is not True
            or payload.get("operator_supplied_local_asset_only") is not False
            or payload.get("runtime_download_allowed") is not True
            or payload.get("redistribution_allowed") is not False
            or payload.get("licence_status") != "DISPUTED_UNRESOLVED"
        ):
            raise ValueError("Reviewed identity model manifest changed")


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
        download=ModelDownload(
            repository="conservationxlabs/miewid-msv3",
            revision=MIEWID_PYTORCH_REVISION,
            filename="model.safetensors",
        ),
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


def _download_checkpoint(asset: ModelAsset, checkpoint: Path) -> None:
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    logger.info(
        "Downloading pinned identity model %s to %s",
        asset.download.reference,
        checkpoint,
    )
    try:
        with requests.get(
            asset.download.url,
            allow_redirects=True,
            headers={
                "Accept-Encoding": "identity",
                "User-Agent": MODEL_DOWNLOAD_USER_AGENT,
            },
            stream=True,
            timeout=MODEL_DOWNLOAD_TIMEOUT_SECONDS,
        ) as response:
            response.raise_for_status()
            response_chain = [*response.history, response]
            if any(
                urlparse(item.url).scheme.lower() != "https" for item in response_chain
            ):
                raise ModelDownloadError("Pinned identity model download left HTTPS")
            content_length = response.headers.get("Content-Length")
            if content_length is not None:
                try:
                    declared_size = int(content_length)
                except ValueError as error:
                    raise ModelDownloadError(
                        "Pinned identity model returned an invalid content length"
                    ) from error
                if declared_size != asset.size_bytes:
                    raise ModelDownloadError(
                        "Downloaded identity checkpoint byte size changed"
                    )

            digest = hashlib.sha256()
            downloaded_size = 0
            with tempfile.NamedTemporaryFile(
                dir=checkpoint.parent,
                mode="wb",
                prefix=f".{checkpoint.name}.",
                suffix=".partial",
                delete=False,
            ) as stream:
                temporary = Path(stream.name)
                for chunk in response.iter_content(
                    chunk_size=MODEL_DOWNLOAD_CHUNK_BYTES
                ):
                    if not chunk:
                        continue
                    downloaded_size += len(chunk)
                    if downloaded_size > asset.size_bytes:
                        raise ModelDownloadError(
                            "Downloaded identity checkpoint exceeds reviewed size"
                        )
                    digest.update(chunk)
                    stream.write(chunk)
                stream.flush()
                os.fsync(stream.fileno())

        if downloaded_size != asset.size_bytes:
            raise ModelDownloadError("Downloaded identity checkpoint byte size changed")
        if digest.hexdigest() != asset.sha256:
            raise ModelDownloadError("Downloaded identity checkpoint checksum changed")
        if temporary is None:
            raise ModelDownloadError(
                "Pinned identity model temporary file was not created"
            )
        os.replace(temporary, checkpoint)
        temporary = None
        logger.info("Installed pinned identity model at %s", checkpoint)
    except ModelDownloadError:
        raise
    except (OSError, requests.RequestException) as error:
        raise ModelDownloadError(
            f"Could not download pinned identity model {asset.download.reference}"
        ) from error
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                logger.warning(
                    "Could not remove incomplete identity model download %s",
                    temporary,
                    exc_info=True,
                )


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
