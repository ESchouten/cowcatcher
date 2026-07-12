import itertools
import json
import logging
import os
import sys
import tempfile
from dataclasses import dataclass, field
from importlib import util
from pathlib import Path
from typing import Any

from aidetector.utils.config import Config, config_list
from aidetector.utils.version import TYPE
from aidetector.utils.winml import WinML

LOGGER = logging.getLogger(__name__)


@dataclass
class OrtState:
    devices: list[tuple[Any, dict]] = field(default_factory=list)
    providers: list[str] = field(default_factory=list)
    dll_dirs: list[Any] = field(default_factory=list)
    is_available: bool = False

    @property
    def names(self) -> list[str]:
        return (
            [device.ep_name for device, _ in self.devices]
            if self.devices
            else self.providers
        )


_STATE = OrtState()


def _should_auto_install_windows_ml_ep(config: Config) -> bool:
    no_auto_install = ["GITHUB_ACTIONS", "CI"]
    return (
        not any(_read_env_bool(name) is True for name in no_auto_install)
        and sys.platform == "win32"
        and TYPE == "windowsml"
        and config.onnx.winml
    )


def should_rect() -> bool:
    return _STATE.names[0] != "NvTensorRTRTXExecutionProvider" if _STATE.names else True


def should_half() -> bool:
    return _STATE.names[0] != "OpenVINOExecutionProvider" if _STATE.names else True


def _patch_ultralytics_requirements() -> None:
    try:
        from ultralytics.utils import checks
    except ImportError:
        return

    _original_check_requirements = checks.check_requirements

    def _check_requirements(requirements=(), **kwargs):
        def should_skip(requirement: str) -> bool:
            return (
                "onnxruntime-gpu" in requirement
                or "onnxruntime" in requirement
                or "onnx" in requirement
            )

        if isinstance(requirements, (list, tuple)):
            requirements = [r for r in requirements if not should_skip(r)]
        elif isinstance(requirements, str) and should_skip(requirements):
            return True

        return _original_check_requirements(requirements=requirements, **kwargs)

    checks.check_requirements = _check_requirements  # ty: ignore[invalid-assignment]

    for module_name in ("ultralytics.nn.autobackend", "ultralytics.engine.exporter"):
        module = sys.modules.get(module_name)
        if module is not None and hasattr(module, "check_requirements"):
            setattr(module, "check_requirements", _check_requirements)


def _candidate_windows_dll_dirs(root: Path) -> list[Path]:
    if not root.exists() or not root.is_dir():
        return []

    candidates = [root, root / "bin", root / "lib"]
    for child in root.iterdir():
        if child.is_dir():
            candidates.append(child)
            candidates.extend(subdir for subdir in child.iterdir() if subdir.is_dir())
    return [path for path in candidates if path.is_dir() and any(path.glob("*.dll"))]


def _add_windows_dll_directories() -> None:
    if (
        sys.platform != "win32"
        or not hasattr(os, "add_dll_directory")
        or TYPE not in ("cuda", "tensorrt")
    ):
        return

    seen: set[str] = set()
    for package_name in (
        "nvidia.cuda_nvrtc",
        "nvidia.cuda_runtime",
        "nvidia.cufft",
        "nvidia.curand",
        "nvidia.cudnn",
        "tensorrt",
        "tensorrt_bindings",
        "tensorrt_libs",
        "tensorrt_cu12_bindings",
        "tensorrt_cu12_libs",
    ):
        try:
            spec = util.find_spec(package_name)
        except ModuleNotFoundError:
            LOGGER.info("Module %s not found", package_name)
            continue
        if spec is None or not spec.submodule_search_locations:
            continue
        for location in spec.submodule_search_locations:
            root = Path(location)
            if not root.exists() or not root.is_dir():
                LOGGER.info("Root %s not found", root)
                continue
            for dll_dir in _candidate_windows_dll_dirs(root):
                dll_dir_str = str(dll_dir)
                normalized = os.path.normcase(os.path.normpath(dll_dir_str))
                if normalized in seen:
                    continue
                seen.add(normalized)
                _STATE.dll_dirs.append(os.add_dll_directory(dll_dir_str))
                os.environ["PATH"] = (
                    dll_dir_str + os.pathsep + os.environ.get("PATH", "")
                )


def setup_ort(config: Config) -> bool:
    try:
        import onnxruntime as ort

        if _STATE.is_available:
            return True

        LOGGER.info("Setup ORT")

        _patch_ultralytics_requirements()
        _add_windows_dll_directories()

        if TYPE in ("cuda", "tensorrt") and hasattr(ort, "preload_dlls"):
            ort.preload_dlls(directory="")

        registered_winml_providers: list[str] = []
        if _should_auto_install_windows_ml_ep(config):
            LOGGER.info("Registering WinML execution provider...")
            try:
                registered_winml_providers = (
                    WinML().register_execution_providers_to_ort()
                )
                LOGGER.info(
                    "Registered WinML providers: %s", registered_winml_providers
                )
            except Exception as e:
                LOGGER.warning("Failed to register WinML execution provider: %s", e)
        else:
            LOGGER.info("Not registering WinML execution provider.")

        def init_devices_and_providers(
            config: Config, registered_winml_providers: list[str]
        ):
            selected_devices = [
                ep_device
                for ep_device in ort.get_ep_devices()
                if ep_device.ep_name in registered_winml_providers
                and (
                    config.onnx.provider is None
                    or config.onnx.provider == ep_device.ep_name
                )
            ]
            _STATE.devices = get_devices(config, selected_devices)
            _STATE.providers = (
                ort.get_available_providers()
                if config.onnx.provider is None
                else [config.onnx.provider]
            )

        init_devices_and_providers(config, registered_winml_providers)

        _original_InferenceSession = ort.InferenceSession

        def InferenceSession(
            path_or_bytes, sess_options=None, providers=None, **kwargs
        ):
            if _STATE.devices:
                LOGGER.info(
                    "Selected ORT EP devices: %s",
                    [device.ep_name for device, _ in _STATE.devices],
                )
                if sess_options is None:
                    sess_options = ort.SessionOptions()

                for device, options in _STATE.devices:
                    sess_options.add_provider_for_devices([device], options)

                return _original_InferenceSession(
                    path_or_bytes, sess_options=sess_options, **kwargs
                )

            LOGGER.info("ORT providers configured for session: %s", _STATE.providers)
            configured_providers = [
                (provider, get_provider_options(config, provider))
                for provider in _STATE.providers
            ]
            return _original_InferenceSession(
                path_or_bytes,
                sess_options=sess_options,
                providers=configured_providers,
                **kwargs,
            )

        ort.InferenceSession = InferenceSession  # ty: ignore[invalid-assignment]
        _STATE.is_available = True

    except Exception as e:
        LOGGER.error("Failed to setup ORT: %s", e)
        return False
    return True


def get_device_options(config: Config, device) -> dict:
    if device.ep_name == "OpenVINOExecutionProvider":
        return _openvino_options(device)
    elif device.ep_name == "NvTensorRTRTXExecutionProvider":
        return _nvtensorrtx_options(config)
    return {}


def get_provider_options(config: Config, provider: str) -> dict:
    if provider == "CoreMLExecutionProvider":
        return _coreml_options()
    return {}


def sort_devices_by_provider(devices: list) -> list:
    devices_by_provider: dict[str, list] = {
        provider_name: list(devices)
        for provider_name, devices in itertools.groupby(
            sorted(devices, key=lambda ep_device: ep_device.ep_name),
            key=lambda ep_device: ep_device.ep_name,
        )
    }
    if "OpenVINOExecutionProvider" in devices_by_provider:
        for device in devices_by_provider.get("OpenVINOExecutionProvider", []):
            if str(device.device.type).endswith("GPU"):
                devices_by_provider["OpenVINOExecutionProvider"] = [device]
                break
        devices_by_provider["OpenVINOExecutionProvider"] = devices_by_provider[
            "OpenVINOExecutionProvider"
        ][:1]

    devices_minus_openvino = [
        d for d in devices_by_provider.items() if d[0] != "OpenVINOExecutionProvider"
    ]
    sorted_devices = []
    for provider_name, devices in devices_minus_openvino:
        sorted_devices.extend(devices)

    if "OpenVINOExecutionProvider" in devices_by_provider:
        sorted_devices.extend(devices_by_provider["OpenVINOExecutionProvider"])
    return sorted_devices


def get_devices(config: Config, devices: list) -> list[tuple]:
    sorted_devices = sort_devices_by_provider(devices)
    device_options = [
        (device, get_device_options(config, device)) for device in sorted_devices
    ]
    return device_options


def _nvtensorrtx_options(config: Config):
    input_name = "images"
    colors = 3
    yolo_configs = [
        detector.yolo for detector in config.detectors if detector.yolo is not None
    ]
    if not yolo_configs:
        return {}

    size_min = min(yolo.imgsz for yolo in yolo_configs)
    size_max = max(yolo.imgsz for yolo in yolo_configs)
    streams_max = max(
        [1]
        + [
            len(config_list(detector.detection.source))
            for detector in config.detectors
            if detector.yolo is not None
        ]
    )
    streams_max = max(streams_max, 1)
    return {
        "nv_profile_min_shapes": f"{input_name}:1x{colors}x{size_min}x{size_min}",
        "nv_profile_opt_shapes": f"{input_name}:{streams_max}x{colors}x{size_max}x{size_max}",
        "nv_profile_max_shapes": f"{input_name}:{streams_max}x{colors}x{size_max}x{size_max}",
    }


def _openvino_options(device):
    device_type = "CPU"
    if str(device.device.type).endswith("GPU"):
        device_type = "GPU"
    precision_hint = "f32"

    config_dir = Path(tempfile.gettempdir()) / "ai-detector" / "openvino"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{device_type.lower()}-{precision_hint}.json"
    config_path.write_text(
        json.dumps({device_type: {"INFERENCE_PRECISION_HINT": precision_hint}}),
        encoding="utf-8",
    )
    return {
        "load_config": str(config_path),
    }


def _coreml_options() -> dict[str, str]:
    return {
        "ModelFormat": "MLProgram",
        "SpecializationStrategy": "FastPrediction",
        "EnableOnSubgraphs": "1",
        "AllowLowPrecisionAccumulationOnGPU": "1",
    }


def _read_env_bool(name: str) -> bool | None:
    value = os.getenv(name)
    if value is None:
        return None

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return None
