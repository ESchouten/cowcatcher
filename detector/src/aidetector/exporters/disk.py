import json
from dataclasses import asdict, field
from pathlib import Path

from aidetector.exporters.exporter import Exporter
from aidetector.media.video import generate_mp4, get_image, get_plot
from aidetector.utils.config import (
    Detection,
    DiskConfig,
    get_date_path,
    get_timestamped_filename,
    max_confidence,
)
from pydantic.dataclasses import dataclass


class DiskExporter(Exporter[DiskConfig]):
    directory: Path | None

    def __init__(self, config: DiskConfig):
        super().__init__(config)
        self.directory = Path("detections") / config.directory if config.directory else None
        if self.directory:
            self.directory.mkdir(parents=True, exist_ok=True)

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
        directory = self.directory or Path("detections") / confidence_max[0]
        directory.mkdir(parents=True, exist_ok=True)

        timestamped_directory = directory / subfolder / timestamp
        timestamped_directory.mkdir(parents=True, exist_ok=True)
        if self.config.strategy == "ALL":
            for result in detections:
                image_name = get_timestamped_filename(result)
                image_path = timestamped_directory / image_name
                with open(image_path, "wb") as f:
                    f.write(get_image(result.images.jpg))
        if best_detection:
            image_path = timestamped_directory / "best.jpg"
            with open(image_path, "wb") as f:
                f.write(get_image(get_plot(best_detection)))
            clean_image_path = timestamped_directory / "clean.jpg"
            with open(clean_image_path, "wb") as f:
                f.write(get_image(best_detection.images.jpg))
        video = generate_mp4(detections, padding=self.config.crop_padding)
        if video:
            video_path = timestamped_directory / "video.mp4"
            with open(video_path, "wb") as f:
                f.write(video)
        crop_region = best_detection.images.crop_region
        metadata: Metadata = Metadata(
            timestamp=timestamp,
            validated=validated,
            confidence=max_confidence(best_detection.confidence),
            confidences=best_detection.confidence,
            identity=asdict(best_detection.identities[0])
            if best_detection.identities
            else None,
            identities=[asdict(identity) for identity in best_detection.identities],
            detections=len(detections),
            start=detections[0].date.isoformat(),
            end=detections[-1].date.isoformat(),
            duration=(detections[-1].date - detections[0].date).total_seconds(),
            crop={
                "x1": crop_region.x1,
                "y1": crop_region.y1,
                "x2": crop_region.x2,
                "y2": crop_region.y2,
            }
            if crop_region
            else None,
            crops=[
                {
                    "x1": obj.crop.x1,
                    "y1": obj.crop.y1,
                    "x2": obj.crop.x2,
                    "y2": obj.crop.y2,
                    "label": obj.crop.label,
                    "confidence": obj.crop.confidence,
                    "identity": asdict(obj.identity) if obj.identity else None,
                }
                for obj in best_detection.images.objects
            ],
        )
        metadata_path = timestamped_directory / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(asdict(metadata), f)


@dataclass
class Metadata:
    timestamp: str
    validated: bool | None
    confidence: float
    confidences: dict[str, float]
    identity: dict[str, str | float | None] | None
    identities: list[dict[str, str | float | None]]
    detections: int
    start: str
    end: str
    duration: float
    crop: dict[str, int] | None = None
    crops: list[dict] = field(default_factory=list)
