import json
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic.dataclasses import dataclass


@dataclass
class Detection:
    date: datetime
    jpg: bytes
    confidence: float


def get_timestamped_filename(detection: Detection) -> str:
    rounded_confidence = round(detection.confidence, 3)
    timestamp = get_date_path(detection, "milliseconds")
    return f"{timestamp}_{rounded_confidence}.jpg"


def get_date_path(detection: Detection, timespec: Literal["seconds", "milliseconds"]) -> str:
    return detection.date.isoformat(timespec=timespec).replace(":", "-")


@dataclass
class CollectionConfig:
    time_seconds: int
    frames_min: int
    confidence_threshold: float


@dataclass
class DetectorConfig:
    collection: CollectionConfig
    model_url: str
    sources: list[str]
    save_directory: Path | None = None
    telegram_chat_id: str | None = None


@dataclass
class Config:
    detectors: list[DetectorConfig]
    telegram_bot_token: str | None = None
    telegram_chat_id: str | None = None


config_json = json.load(open("config.json"))
if config_json is None:
    raise ValueError("Config file is empty or not found.")
config = Config(**config_json)
