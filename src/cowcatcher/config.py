from pathlib import Path
from pydantic.dataclasses import dataclass

@dataclass
class DetectorConfig:
    collection_seconds: int
    confidence_threshold: float
    model_url: Path
    save_directory: Path | None
    sources: list[Path]

@dataclass
class Config:
    detectors: list[DetectorConfig]