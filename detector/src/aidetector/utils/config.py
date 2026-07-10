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
HttpMethod = Literal["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"]


@dataclass
class IdentityResult:
    identity: str
    similarity: float


@dataclass
class Crop:
    x1: int
    y1: int
    x2: int
    y2: int
    label: str | None = None
    confidence: float | None = None
    track_id: int | None = None
    identity: IdentityResult | None = None


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class ImageSet:
    jpg: ndarray
    crops: list[Crop] = field(default_factory=list)

    @property
    def crop_region(self) -> Crop | None:
        if not self.crops:
            return None
        if len(self.crops) == 1:
            return self.crops[0]
        return Crop(
            min(crop.x1 for crop in self.crops),
            min(crop.y1 for crop in self.crops),
            max(crop.x2 for crop in self.crops),
            max(crop.y2 for crop in self.crops),
        )


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class Detection:
    date: datetime
    images: ImageSet
    confidence: Confidence

    @property
    def identities(self) -> list[IdentityResult]:
        identities = []
        seen = set()
        for crop in self.images.crops:
            if crop.identity is None or crop.identity.identity in seen:
                continue
            identities.append(crop.identity)
            seen.add(crop.identity.identity)
        return identities


@dataclass(kw_only=True)
class ModelConfig:
    confidence: float | Confidence = 0
    time_max: int = 60
    timeout: int = 5
    cooldown: float | dict[str, float] = 0
    include_trailing_time: int = 1
    frames_min: int = field(default_factory=_default_frames_min)


@dataclass(kw_only=True)
class YoloConfig(ModelConfig):
    model: str
    task: Literal["detect", "segment"] = "detect"
    tracking: bool = False
    imgsz: int = 640


@dataclass(kw_only=True)
class DazzleCowConfig(ModelConfig):
    model: Path
    gallery: Path
    owl_model: str = "google/owlv2-large-patch14-ensemble"
    sam_model: str = "sam2.1_l.pt"
    owl_interval: float = 1
    prompt: str = "cow"
    confidence: float = 0.3
    match_threshold: float = 0.75
    match_margin: float = 0
    neighbors: int = 5
    min_area_ratio: float = 0.025
    max_area_ratio: float = 0.075
    nms_iou: float = 0.5
    track_samples: int = 5
    track_iou: float = 0.2
    track_max_age: int = 10
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    frames_min: int = 1


@dataclass(kw_only=True)
class DetectionConfig:
    source: str | list[str]
    interval: float = 0
    frame_retention: int = 15
    frames_width: int = 1280


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
class HttpConfig:
    url: str
    method: HttpMethod = "GET"
    timeout: int | None = None
    headers: dict[str, str] | None = field(default=None, repr=False)
    body: str | None = None


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
class WebhookConfig(ExporterConfig, HttpConfig):
    method: HttpMethod = "POST"
    token: str | None = field(default=None, repr=False)
    data_type: Literal["binary", "base64", "none"] = "binary"
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


@dataclass(kw_only=True)
class SSEConfig(ExporterConfig):
    port: int = 8765
    endpoint: str | None = None


@dataclass
class ExportersConfig:
    disk: DiskConfig | list[DiskConfig] | None = None
    telegram: ChatConfig | list[ChatConfig] | None = None
    webhook: WebhookConfig | list[WebhookConfig] | None = None
    sse: SSEConfig | list[SSEConfig] | None = None


@dataclass(kw_only=True)
class HealthcheckConfig(HttpConfig):
    interval: int = 60
    timeout: int = 5


@dataclass
class DetectorConfig:
    detection: DetectionConfig
    yolo: YoloConfig | None = None
    dazzlecow: DazzleCowConfig | None = None
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


def get_date_path(detection: Detection, timespec: Literal["seconds", "milliseconds"]) -> str:
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


def load_config() -> Config:
    config_path = Path("config.json")
    if not config_path.exists():
        template = get_template()
        if template:
            with open(config_path, "w") as f:
                json.dump(template, f, indent=4)
            logger.warning(f"Created {config_path} from template. Please edit the configuration before running.")
            raise FileNotFoundError(f"Configure before running: {config_path}")
        else:
            logger.error(f"Configuration file not found: {config_path}")
            logger.error("Create a config.json file. See: https://github.com/ESchouten/ai-detector")
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
        raise ValueError(f"Configuration validation failed for {config_path}:\n{format_validation_errors(e)}")


config = load_config()
