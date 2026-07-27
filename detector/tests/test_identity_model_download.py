from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator

import pytest
import requests
from aidetector.reid import models as model_registry
from aidetector.reid.models import (
    MIEWID_DUAL_CROP_V1,
    MODEL_DOWNLOAD_CHUNK_BYTES,
    MODEL_DOWNLOAD_TIMEOUT_SECONDS,
    MODEL_DOWNLOAD_USER_AGENT,
    MODEL_REGISTRY,
    ModelAsset,
    ModelDownloadError,
)


class FakeResponse:
    def __init__(
        self,
        *,
        url: str,
        chunks: list[bytes],
        content_length: int | None,
        history: list[str] | None = None,
        stream_error: requests.RequestException | None = None,
    ):
        self.url = url
        self.chunks = chunks
        self.headers = (
            {"Content-Length": str(content_length)}
            if content_length is not None
            else {}
        )
        self.history = [SimpleNamespace(url=item) for item in (history or [])]
        self.stream_error = stream_error

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *_args) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, *, chunk_size: int) -> Iterator[bytes]:
        assert chunk_size == MODEL_DOWNLOAD_CHUNK_BYTES
        for chunk in self.chunks:
            yield chunk
        if self.stream_error is not None:
            raise self.stream_error


def fixture_asset(
    payload: bytes,
    *,
    sha256: str | None = None,
    size_bytes: int | None = None,
) -> ModelAsset:
    return replace(
        MODEL_REGISTRY[MIEWID_DUAL_CROP_V1],
        checkpoint="models/miewid/download-fixture.safetensors",
        sha256=sha256 or hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload) if size_bytes is None else size_bytes,
    )


def partial_files(target: Path) -> list[Path]:
    return list(target.parent.glob(f".{target.name}.*.partial"))


def test_missing_checkpoint_downloads_atomically_from_pinned_url(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"reviewed-safe-tensors"
    asset = fixture_asset(payload)
    target = tmp_path / asset.checkpoint
    request: dict[str, object] = {}

    def fake_get(url: str, **kwargs) -> FakeResponse:
        request["url"] = url
        request["kwargs"] = kwargs
        return FakeResponse(
            url=url,
            chunks=[payload[:8], b"", payload[8:]],
            content_length=len(payload),
        )

    real_replace = model_registry.os.replace

    def observed_replace(source: Path, destination: Path) -> None:
        assert not target.exists()
        assert Path(source).name.endswith(".partial")
        assert Path(destination) == target
        real_replace(source, destination)

    monkeypatch.setattr(model_registry.requests, "get", fake_get)
    monkeypatch.setattr(model_registry.os, "replace", observed_replace)

    model_registry._download_checkpoint(asset, target)

    assert target.read_bytes() == payload
    assert partial_files(target) == []
    assert request == {
        "url": asset.download.url,
        "kwargs": {
            "allow_redirects": True,
            "headers": {
                "Accept-Encoding": "identity",
                "User-Agent": MODEL_DOWNLOAD_USER_AGENT,
            },
            "stream": True,
            "timeout": MODEL_DOWNLOAD_TIMEOUT_SECONDS,
        },
    }


@pytest.mark.parametrize(
    ("asset", "response", "message"),
    [
        (
            fixture_asset(b"expected"),
            FakeResponse(
                url=MODEL_REGISTRY[MIEWID_DUAL_CROP_V1].download.url,
                chunks=[b"short"],
                content_length=5,
            ),
            "byte size changed",
        ),
        (
            fixture_asset(b"expected", sha256="0" * 64),
            FakeResponse(
                url=MODEL_REGISTRY[MIEWID_DUAL_CROP_V1].download.url,
                chunks=[b"expected"],
                content_length=8,
            ),
            "checksum changed",
        ),
        (
            fixture_asset(b"four", size_bytes=4),
            FakeResponse(
                url=MODEL_REGISTRY[MIEWID_DUAL_CROP_V1].download.url,
                chunks=[b"123", b"456"],
                content_length=None,
            ),
            "exceeds reviewed size",
        ),
    ],
)
def test_invalid_download_never_installs_or_leaves_partial_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    asset: ModelAsset,
    response: FakeResponse,
    message: str,
) -> None:
    target = tmp_path / asset.checkpoint
    monkeypatch.setattr(
        model_registry.requests,
        "get",
        lambda *_args, **_kwargs: response,
    )

    with pytest.raises(ModelDownloadError, match=message):
        model_registry._download_checkpoint(asset, target)

    assert not target.exists()
    assert partial_files(target) == []


def test_interrupted_download_is_cleaned_up_and_reported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"expected"
    asset = fixture_asset(payload)
    target = tmp_path / asset.checkpoint
    response = FakeResponse(
        url=asset.download.url,
        chunks=[payload[:3]],
        content_length=None,
        stream_error=requests.ConnectionError("connection lost"),
    )
    monkeypatch.setattr(
        model_registry.requests,
        "get",
        lambda *_args, **_kwargs: response,
    )

    with pytest.raises(ModelDownloadError, match="Could not download"):
        model_registry._download_checkpoint(asset, target)

    assert not target.exists()
    assert partial_files(target) == []


def test_download_rejects_non_https_redirects_before_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"expected"
    asset = fixture_asset(payload)
    target = tmp_path / asset.checkpoint
    response = FakeResponse(
        url=asset.download.url,
        chunks=[payload],
        content_length=len(payload),
        history=["http://untrusted.invalid/model.safetensors"],
    )
    monkeypatch.setattr(
        model_registry.requests,
        "get",
        lambda *_args, **_kwargs: response,
    )

    with pytest.raises(ModelDownloadError, match="left HTTPS"):
        model_registry._download_checkpoint(asset, target)

    assert not target.exists()
    assert partial_files(target) == []
