import json
import logging
from dataclasses import field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import requests
from aidetector.utils.version import REF_NAME
from numpy import ndarray
from pydantic import ConfigDict, ValidationError
from pydantic.dataclasses import dataclass

logger = logging.getLogger(__name__)


def _default_frames_min() -> int:
    try:
        import torch

        if torch.cuda.is_available():
            return 6
    except ImportError:
        pass
    return 3


Confidence = dict[str, float]


@dataclass
class Crop:
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class ImageSet:
    jpg: ndarray
    plot: ndarray | None
    crop: Crop | None


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class Detection:
    date: datetime
    images: ImageSet
    confidence: Confidence


@dataclass(kw_only=True)
class YoloConfig:
    model: str
    confidence: float | Confidence = 0
    time_max: int = 60
    timeout: int = 5
    cooldown: float | dict[str, float] = 0
    include_trailing_time: int = 1
    frames_min: int = field(default_factory=_default_frames_min)
    imgsz: int = 640
    strategy: Literal["LATEST", "ALL"] = "LATEST"


@dataclass(kw_only=True)
class DetectionConfig:
    source: str | list[str]
    interval: float = 0
    frame_retention: int = 15
    frames_width: int = 1080


@dataclass(kw_only=True)
class VLMConfig:
    prompt: str
    model: str | list[str]
    key: str | None = field(default=None, repr=False)
    url: str | None = None
    strategy: Literal["IMAGE", "VIDEO"] = "VIDEO"
    crop_padding: float = 0.1


@dataclass(kw_only=True)
class ExporterConfig:
    confidence: float | Confidence | None = None
    crop_padding: float = 0.1
    export_rejected: bool = False


@dataclass(kw_only=True)
class ChatConfig(ExporterConfig):
    token: str = field(repr=False)
    chat: str
    alert_every: int = 1
    include_image: bool = False
    include_plot: bool = False
    include_crop: bool = False
    include_video: bool = True
    video_width: int | None = 1280
    video_crf: int = 28


@dataclass(kw_only=True)
class WebhookConfig(ExporterConfig):
    url: str
    token: str | None = field(default=None, repr=False)
    data_type: Literal["binary", "base64"] = "binary"
    data_max: int | None = None
    include_image: bool = False
    include_plot: bool = False
    include_crop: bool = True
    include_video: bool = False
    video_width: int | None = 1280
    video_crf: int = 28


@dataclass(kw_only=True)
class DiskConfig(ExporterConfig):
    directory: Path | None = None
    strategy: Literal["ALL", "BEST"] = "BEST"
    export_rejected: bool = True


@dataclass
class ExportersConfig:
    disk: DiskConfig | list[DiskConfig] | None = None
    telegram: ChatConfig | list[ChatConfig] | None = None
    webhook: WebhookConfig | list[WebhookConfig] | None = None


@dataclass(kw_only=True)
class HealthcheckConfig:
    url: str
    method: Literal["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"] = "GET"
    interval: int = 60
    timeout: int = 5
    headers: dict[str, str] | None = field(default=None, repr=False)
    body: str | None = None


@dataclass
class DetectorConfig:
    detection: DetectionConfig
    yolo: YoloConfig | None = None
    vlm: VLMConfig | list[VLMConfig] | None = None
    exporters: ExportersConfig | None = None


@dataclass(kw_only=True)
class OnnxConfig:
    provider: str | None = None
    winml: bool = True
    opset: int = 20


@dataclass
class Config:
    detectors: list[DetectorConfig]
    onnx: OnnxConfig = field(default_factory=OnnxConfig)
    health: HealthcheckConfig | None = None


template_url = f"https://raw.githubusercontent.com/ESchouten/ai-detector/{REF_NAME}/config/config.template.json"
schema_url = f"https://raw.githubusercontent.com/ESchouten/ai-detector/{REF_NAME}/config/config.schema.json"


def get_template() -> Any | None:
    try:
        template = requests.get(template_url).json()
        template["$schema"] = schema_url
        return template
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch template from {template_url}: {e}")
        return None


def get_timestamped_filename(detection: Detection) -> str:
    rounded_confidence = round(max_confidence(detection.confidence), 3)
    timestamp = get_date_path(detection, "milliseconds")
    return f"{timestamp}_{rounded_confidence}.jpg"


def get_date_path(
    detection: Detection, timespec: Literal["seconds", "milliseconds"]
) -> str:
    return detection.date.isoformat(timespec=timespec).replace(":", "-")


def min_confidence(confidence: float | Confidence | None) -> float:
    if confidence is None or not confidence:
        return 0
    if isinstance(confidence, dict):
        return min(float(value) for value in confidence.values())
    return float(confidence)


def max_confidence(confidence: float | Confidence | None) -> float:
    if confidence is None or not confidence:
        return 0
    if isinstance(confidence, dict):
        return max(float(value) for value in confidence.values())
    return float(confidence)


def confidence_matches(
    value: dict[str, float],
    threshold: float | dict[str, float],
) -> bool:
    if isinstance(threshold, dict):
        return any(
            class_id in value and float(value[class_id]) >= float(class_threshold)
            for class_id, class_threshold in threshold.items()
        )
    else:
        return max_confidence(value) >= threshold


def matching_confidences(
    value: dict[str, float],
    threshold: float | dict[str, float],
) -> list[str]:
    return [
        class_name
        for class_name, confidence in value.items()
        if confidence_matches({class_name: confidence}, threshold)
    ]


def format_validation_errors(error: ValidationError) -> str:
    messages = []
    for err in error.errors():
        location = " -> ".join(str(loc) for loc in err["loc"])
        msg = err["msg"]
        messages.append(f"  • {location}: {msg}")
    return "\n".join(messages)


def load_config(config_path: Path = Path("config.json")) -> Config:
    if not config_path.exists():
        template = get_template()
        if template:
            with open(config_path, "w") as f:
                json.dump(template, f, indent=4)
            logger.warning(
                f"Created {config_path} from template. Please edit the configuration before running."
            )
            raise FileNotFoundError(f"Configure before running: {config_path}")
        else:
            logger.error(f"Configuration file not found: {config_path}")
            logger.error(
                "Create a config.json file. See: https://github.com/ESchouten/ai-detector"
            )
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path) as f:
            config_json = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {config_path}: {e}")
        raise ValueError(f"Invalid JSON in {config_path}: {e}")

    if config_json is None:
        logger.error(f"Config file is empty: {config_path}")
        raise ValueError(f"Config file is empty: {config_path}")

    try:
        config_json["$schema"] = schema_url
        with open(config_path, "w") as f:
            json.dump(config_json, f, indent=4)
    except Exception as e:
        logger.warning(f"Failed to update schema in {config_path}: {e}")

    try:
        return Config(**config_json)
    except ValidationError as e:
        logger.error(f"Configuration validation failed for {config_path}:")
        logger.error(format_validation_errors(e))
        raise ValueError(
            f"Configuration validation failed for {config_path}:\n{format_validation_errors(e)}"
        )


config = load_config()
