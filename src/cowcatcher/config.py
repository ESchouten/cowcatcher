from pathlib import Path

from pydantic.dataclasses import dataclass


@dataclass
class DetectorConfig:
    collection_seconds: int
    confidence_threshold: float
    model_url: str
    sources: list[str]
    save_directory: Path | None = None
    telegram_chat_id: str | None = None


@dataclass
class Config:
    detectors: list[DetectorConfig]
    telegram_bot_token: str | None = None
    telegram_chat_id: str | None = None
