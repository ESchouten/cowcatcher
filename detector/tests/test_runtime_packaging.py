from __future__ import annotations

import re
import tomllib
from pathlib import Path

from aidetector.reid.models import MODEL_REGISTRY

DETECTOR_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = DETECTOR_ROOT.parent
SOURCE_ROOT = DETECTOR_ROOT / "src/aidetector"
PYPROJECT_PATH = DETECTOR_ROOT / "pyproject.toml"

IDENTITY_RUNTIME_REQUIREMENTS = {
    "requests",
    "safetensors",
    "timm",
    "torch",
    "torchvision",
}
PRESERVED_VLM_REQUIREMENTS = {"litellm", "tiktoken"}
RESEARCH_ONLY_REQUIREMENTS = {
    "fastapi",
    "huggingface-hub",
    "kornia",
    "lightglue",
    "pandas",
    "scikit-learn",
    "scipy",
    "transformers",
    "uvicorn",
    "xgboost",
}


def _requirement_name(requirement: str) -> str:
    return re.split(r"[\s\[<>=@;]", requirement, maxsplit=1)[0].lower()


def _configuration() -> dict:
    with PYPROJECT_PATH.open("rb") as stream:
        return tomllib.load(stream)


def test_runtime_declares_identity_without_research_dependencies() -> None:
    project = _configuration()["project"]
    runtime = {_requirement_name(item) for item in project["dependencies"]}

    assert IDENTITY_RUNTIME_REQUIREMENTS <= runtime
    assert PRESERVED_VLM_REQUIREMENTS <= runtime
    assert runtime.isdisjoint(RESEARCH_ONLY_REQUIREMENTS)
    assert set(project["optional-dependencies"]) == {
        "default",
        "nvidia",
        "windowsml",
    }
    assert project["scripts"] == {
        "main": "aidetector:main",
        "generate-schema": "aidetector.utils.generate_schema:main",
    }


def test_production_model_download_uses_only_pinned_safe_weights() -> None:
    production_source = "\n".join(
        path.read_text(encoding="utf-8") for path in SOURCE_ROOT.rglob("*.py")
    ).lower()

    assert "huggingface_hub" not in production_source
    assert "hf_hub_download" not in production_source
    assert ".from_pretrained(" not in production_source
    assert "trust_remote_code" not in production_source
    assert "torch.load(" not in production_source
    for asset in MODEL_REGISTRY.values():
        assert "://" not in asset.checkpoint
        assert asset.checkpoint.startswith("models/")
        assert asset.download.revision == asset.immutable_revision
        assert asset.download.filename.endswith(".safetensors")
        assert asset.download.url.startswith("https://huggingface.co/")
        assert f"/resolve/{asset.immutable_revision}/" in asset.download.url
        manifest = SOURCE_ROOT / "reid/model_manifests" / asset.manifest
        assert manifest.is_file()
        assert len(asset.sha256) == 64


def test_release_keeps_vlm_and_bundles_identity_package_data() -> None:
    workflow = (REPOSITORY_ROOT / ".github/workflows/detector.yaml").read_text(
        encoding="utf-8"
    )
    arguments = next(
        line.strip().split()
        for line in workflow.splitlines()
        if "PYINSTALLER_ARGS:" in line
    )
    collected = {
        arguments[index + 1]
        for index, argument in enumerate(arguments[:-1])
        if argument == "--collect-all"
    }
    data_packages = {
        arguments[index + 1]
        for index, argument in enumerate(arguments[:-1])
        if argument == "--collect-data"
    }

    assert {"litellm", "onnx", "onnxruntime", "safetensors", "timm"} <= collected
    assert "aidetector" in data_packages
    assert "--hidden-import=tiktoken_ext.openai_public" in arguments
    assert collected.isdisjoint({"kornia", "lightglue", "transformers", "xgboost"})


def test_container_context_excludes_local_build_artifacts() -> None:
    excluded = {
        line.strip()
        for line in (DETECTOR_ROOT / ".dockerignore")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert {".venv", "build", "dist", "*.spec"} <= excluded
