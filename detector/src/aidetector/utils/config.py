import json
import logging
from dataclasses import field
from pathlib import Path
from typing import Annotated, Any, Literal, TypeVar

import requests
from aidetector.utils.version import REF_NAME
from pydantic import ConfigDict, Field, ValidationError
from pydantic.dataclasses import dataclass

logger = logging.getLogger(__name__)
STRICT_CONFIG = ConfigDict(extra="forbid")
T = TypeVar("T")


class ConfigurationError(ValueError):
    pass


def config_list(value: T | list[T] | None) -> list[T]:
    if value is None:
        return []
    return list(value) if isinstance(value, list) else [value]


HttpMethod = Literal["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"]
NonEmptyString = Annotated[str, Field(min_length=1)]
Probability = Annotated[float, Field(ge=0, le=1)]
FrameMargin = Annotated[float, Field(ge=0, lt=0.5)]
NonNegativeFloat = Annotated[float, Field(ge=0)]
PositiveFloat = Annotated[float, Field(gt=0)]
PositiveInt = Annotated[int, Field(gt=0)]
Crf = Annotated[int, Field(ge=0, le=51)]
Port = Annotated[int, Field(ge=1, le=65535)]
Sources = Annotated[list[NonEmptyString], Field(min_length=1)]
Models = Annotated[list[NonEmptyString], Field(min_length=1)]
Cooldown = NonNegativeFloat | dict[str, NonNegativeFloat]
ConfidenceThreshold = Probability | dict[str, Probability]


@dataclass(config=STRICT_CONFIG, kw_only=True)
class YoloConfig:
    model: NonEmptyString
    task: Literal["detect", "segment"] = "detect"
    confidence: ConfidenceThreshold = 0
    tracking: bool = False
    time_max: NonNegativeFloat = 60
    timeout: NonNegativeFloat = 5
    cooldown: Cooldown = 0
    include_trailing_time: NonNegativeFloat = 1
    frames_min: PositiveInt = 3
    imgsz: PositiveInt = 640
    iou: Probability = 0.7
    tracker: NonEmptyString = "bytetrack.yaml"


@dataclass(config=STRICT_CONFIG, kw_only=True)
class IdentityEnrollmentConfig:
    identity_count: PositiveInt | None = None


@dataclass(config=STRICT_CONFIG, kw_only=True)
class IdentityConfig:
    database: Path
    label: NonEmptyString
    enrollment: IdentityEnrollmentConfig | None = None
    segment_model: NonEmptyString | None = None
    imgsz: PositiveInt = 640
    confidence: Probability = 0.25
    match_threshold: Probability = 0.68
    match_margin: Probability = 0.05
    min_area_ratio: Probability = 0
    max_area_ratio: Probability = 1
    margin: FrameMargin = 0
    nms_iou: Probability = 0.7
    track_samples: PositiveInt = 5
    track_max_age: PositiveInt = 10

    def __post_init__(self) -> None:
        if self.min_area_ratio > self.max_area_ratio:
            raise ValueError("Identity minimum area exceeds maximum area")


@dataclass(config=STRICT_CONFIG, kw_only=True)
class DetectionConfig:
    source: NonEmptyString | Sources
    interval: NonNegativeFloat = 0
    frame_retention: PositiveInt = 15
    frames_width: PositiveInt = 1280


@dataclass(config=STRICT_CONFIG, kw_only=True)
class VLMConfig:
    prompt: NonEmptyString
    model: NonEmptyString | Models
    key: str | None = field(default=None, repr=False)
    url: str | None = None
    strategy: Literal["IMAGE", "VIDEO"] = "VIDEO"
    crop_padding: NonNegativeFloat = 0.1
    timeout: PositiveFloat = 30


@dataclass(config=STRICT_CONFIG, kw_only=True)
class ExporterConfig:
    confidence: ConfidenceThreshold | None = None
    crop_padding: NonNegativeFloat = 0.1
    export_rejected: bool = False


@dataclass(config=STRICT_CONFIG, kw_only=True)
class HttpConfig:
    url: NonEmptyString
    method: HttpMethod = "GET"
    timeout: PositiveFloat | None = None
    headers: dict[str, str] | None = field(default=None, repr=False)
    body: str | None = None


@dataclass(config=STRICT_CONFIG, kw_only=True)
class MediaExporterConfig(ExporterConfig):
    include_image: bool = False
    include_plot: bool = False
    include_crop: bool = False
    include_video: bool = False
    video_width: PositiveInt | None = 1280
    video_crf: Crf = 28


@dataclass(config=STRICT_CONFIG, kw_only=True)
class ChatConfig(MediaExporterConfig):
    token: NonEmptyString = field(repr=False)
    chat: NonEmptyString
    alert_every: PositiveInt = 1
    timeout: PositiveFloat = 30
    include_video: bool = True


@dataclass(config=STRICT_CONFIG, kw_only=True)
class WebhookConfig(MediaExporterConfig, HttpConfig):
    method: HttpMethod = "POST"
    token: str | None = field(default=None, repr=False)
    data_type: Literal["binary", "base64", "none"] = "binary"
    data_max: PositiveInt | None = None
    include_crop: bool = True


@dataclass(config=STRICT_CONFIG, kw_only=True)
class DiskConfig(ExporterConfig):
    directory: Path | None = None
    strategy: Literal["ALL", "BEST"] = "BEST"
    export_rejected: bool = True


@dataclass(config=STRICT_CONFIG, kw_only=True)
class SSEConfig(ExporterConfig):
    port: Port = 8765
    endpoint: str | None = None


@dataclass(config=STRICT_CONFIG)
class ExportersConfig:
    disk: DiskConfig | list[DiskConfig] | None = None
    telegram: ChatConfig | list[ChatConfig] | None = None
    webhook: WebhookConfig | list[WebhookConfig] | None = None
    sse: SSEConfig | list[SSEConfig] | None = None


@dataclass(config=STRICT_CONFIG, kw_only=True)
class HealthcheckConfig(HttpConfig):
    interval: PositiveFloat = 60
    timeout: PositiveFloat = 5


@dataclass(config=STRICT_CONFIG)
class DetectorConfig:
    detection: DetectionConfig
    yolo: YoloConfig | None = None
    identity: IdentityConfig | None = None
    vlm: VLMConfig | list[VLMConfig] | None = None
    exporters: ExportersConfig | None = None

    def __post_init__(self) -> None:
        if self.identity is not None and self.yolo is None:
            raise ValueError("Identity enrichment requires a YOLO detector")


@dataclass(config=STRICT_CONFIG, kw_only=True)
class OnnxConfig:
    provider: str | None = None
    winml: bool = True
    opset: PositiveInt = 20


@dataclass(config=STRICT_CONFIG)
class Config:
    detectors: Annotated[list[DetectorConfig], Field(min_length=1)]
    onnx: OnnxConfig = field(default_factory=OnnxConfig)
    health: HealthcheckConfig | None = None


TEMPLATE_URL = f"https://raw.githubusercontent.com/ESchouten/ai-detector/{REF_NAME}/config/config.template.json"
SCHEMA_URL = f"https://raw.githubusercontent.com/ESchouten/ai-detector/{REF_NAME}/config/config.schema.json"
DEFAULT_CONFIG_PATH = Path("config.json")


def get_template() -> dict[str, Any] | None:
    try:
        response = requests.get(TEMPLATE_URL, timeout=10)
        response.raise_for_status()
        template = response.json()
        if not isinstance(template, dict):
            raise ValueError("Configuration template must be a JSON object")
        template["$schema"] = SCHEMA_URL
        return template
    except (requests.RequestException, ValueError) as error:
        logger.error("Failed to fetch template from %s: %s", TEMPLATE_URL, error)
        return None


def format_validation_errors(error: ValidationError) -> str:
    messages = []
    for err in error.errors():
        location = " -> ".join(str(loc) for loc in err["loc"])
        msg = err["msg"]
        messages.append(f"  • {location}: {msg}")
    return "\n".join(messages)


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> Config:
    if not config_path.exists():
        template = get_template()
        if template:
            config_path.write_text(json.dumps(template, indent=4) + "\n")
            logger.warning(
                "Created %s from template. Please edit it before running.",
                config_path,
            )
            raise ConfigurationError(f"Configure before running: {config_path}")
        logger.error("Configuration file not found: %s", config_path)
        raise ConfigurationError(f"Configuration file not found: {config_path}")

    try:
        config_json = json.loads(config_path.read_text())
    except json.JSONDecodeError as error:
        raise ConfigurationError(f"Invalid JSON in {config_path}: {error}") from error

    if not isinstance(config_json, dict):
        raise ConfigurationError(f"Configuration must be a JSON object: {config_path}")

    config_json.pop("$schema", None)

    try:
        return Config(**config_json)
    except ValidationError as error:
        details = format_validation_errors(error)
        raise ConfigurationError(
            f"Configuration validation failed for {config_path}:\n{details}"
        ) from error
