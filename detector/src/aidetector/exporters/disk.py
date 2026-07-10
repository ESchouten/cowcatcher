import json
from pathlib import Path

from aidetector.detection.models import Detection
from aidetector.exporters.exporter import Exporter
from aidetector.exporters.metadata import DetectionMetadata
from aidetector.exporters.naming import date_path, timestamped_filename
from aidetector.media.video import generate_mp4, get_image, get_plot
from aidetector.utils.config import DiskConfig


class DiskExporter(Exporter[DiskConfig]):
    directory: Path | None

    def __init__(self, config: DiskConfig):
        super().__init__(config)
        self.directory = (
            Path("detections") / config.directory if config.directory else None
        )
        if self.directory:
            self.directory.mkdir(parents=True, exist_ok=True)

    def _export(
        self,
        best_detection: Detection,
        detections: list[Detection],
        validated: bool | None,
    ) -> None:
        self.logger.info("Saving %d frames to disk", len(detections))
        timestamp = date_path(best_detection, "seconds")
        subfolder = (
            "approved"
            if validated
            else "rejected"
            if validated is False
            else "unvalidated"
        )

        label = max(
            best_detection.confidence.items(),
            key=lambda item: item[1],
            default=("unclassified", 0),
        )[0]
        directory = self.directory or Path("detections") / label
        directory.mkdir(parents=True, exist_ok=True)

        timestamped_directory = directory / subfolder / timestamp
        timestamped_directory.mkdir(parents=True, exist_ok=True)
        if self.config.strategy == "ALL":
            for result in detections:
                image_name = timestamped_filename(result)
                image_path = timestamped_directory / image_name
                with open(image_path, "wb") as f:
                    f.write(get_image(result.images.jpg))
        image_path = timestamped_directory / "best.jpg"
        image_path.write_bytes(get_image(get_plot(best_detection)))
        clean_image_path = timestamped_directory / "clean.jpg"
        clean_image_path.write_bytes(get_image(best_detection.images.jpg))
        video = generate_mp4(detections, padding=self.config.crop_padding)
        if video:
            video_path = timestamped_directory / "video.mp4"
            video_path.write_bytes(video)
        metadata = DetectionMetadata.from_event(
            timestamp,
            best_detection,
            detections,
            validated,
        )
        metadata_path = timestamped_directory / "metadata.json"
        metadata_path.write_text(json.dumps(metadata.as_dict()) + "\n")
