import json
from datetime import datetime
from pathlib import Path

from pydantic.dataclasses import dataclass


@dataclass
class Detection:
    date: datetime
    jpg: bytes
    confidence: float


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
