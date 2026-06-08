import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Literal

from aidetector.exporters.exporter import Exporter
from aidetector.media.video import generate_mp4, get_image
from aidetector.utils.config import (
    Confidence,
    Config,
    Detection,
    DetectorConfig,
    DiskConfig,
    get_date_path,
    get_timestamped_filename,
    max_confidence,
)
from pydantic.dataclasses import dataclass
from typing_extensions import Self


class DiskExporter(Exporter[DiskConfig]):
    directory: Path | None
    strategy: Literal["ALL", "BEST"]

    def __init__(
        self,
        directory: Path | None,
        confidence: float | Confidence,
        export_rejected: bool = True,
        strategy: Literal["ALL", "BEST"] = "BEST",
        crop_padding: float = 0.1,
    ):
        super().__init__(confidence, export_rejected, directory)
        if directory:
            self.directory = Path(os.path.join("detections", directory))
            os.makedirs(self.directory, exist_ok=True)
        self.strategy = strategy
        self.crop_padding = crop_padding

    @classmethod
    def from_config(
        cls, config: Config, detector: DetectorConfig, exporter: DiskConfig
    ) -> Self:
        return cls(
            exporter.directory,
            exporter.confidence or 0,
            exporter.export_rejected,
            exporter.strategy,
            exporter.crop_padding,
        )

    def filtered_export(
        self,
        best_detection: Detection,
        detections: list[Detection],
        validated: bool | None,
    ):
        self.logger.info(f"Saving {len(detections)} photos to disk")
        timestamp = get_date_path(best_detection, "seconds")
        subfolder = (
            "approved"
            if validated
            else "rejected"
            if validated is False
            else "unvalidated"
        )

        confidence_max = max(best_detection.confidence.items(), key=lambda x: x[1])
        directory = self.directory or Path(
            os.path.join("detections", confidence_max[0])
        )
        os.makedirs(directory, exist_ok=True)

        timestamped_directory = os.path.join(directory, subfolder, timestamp)
        os.makedirs(timestamped_directory, exist_ok=True)
        if self.strategy == "ALL":
            for result in detections:
                image_name = get_timestamped_filename(result)
                image_path = os.path.join(timestamped_directory, image_name)
                with open(image_path, "wb") as f:
                    f.write(get_image(result.images.jpg))
        if best_detection:
            image_path = os.path.join(timestamped_directory, "best.jpg")
            with open(image_path, "wb") as f:
                f.write(
                    get_image(
                        best_detection.images.plot
                        if best_detection.images.plot is not None
                        else best_detection.images.jpg
                    )
                )
            clean_image_path = os.path.join(timestamped_directory, "clean.jpg")
            with open(clean_image_path, "wb") as f:
                f.write(get_image(best_detection.images.jpg))
        video = generate_mp4(detections, padding=self.crop_padding)
        if video:
            video_path = os.path.join(timestamped_directory, "video.mp4")
            with open(video_path, "wb") as f:
                f.write(video)
        metadata: Metadata = Metadata(
            timestamp=timestamp,
            validated=validated,
            confidence=max_confidence(best_detection.confidence),
            confidences=best_detection.confidence,
            detections=len(detections),
            start=detections[0].date.isoformat(),
            end=detections[-1].date.isoformat(),
            duration=(detections[-1].date - detections[0].date).total_seconds(),
            crop={
                "x1": best_detection.images.crop.x1,
                "y1": best_detection.images.crop.y1,
                "x2": best_detection.images.crop.x2,
                "y2": best_detection.images.crop.y2,
            }
            if best_detection.images.crop
            else None,
        )
        metadata_path = os.path.join(timestamped_directory, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(asdict(metadata), f)


@dataclass
class Metadata:
    timestamp: str
    validated: bool | None
    confidence: float
    confidences: dict[str, float]
    detections: int
    start: str
    end: str
    duration: float
    crop: dict[str, int] | None = None
