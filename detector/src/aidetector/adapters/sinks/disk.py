import json
import logging
from pathlib import Path

from aidetector.adapters.sinks.metadata import DetectionMetadata
from aidetector.adapters.sinks.naming import date_path, timestamped_filename
from aidetector.media.rendering import frame_jpg
from aidetector.pipeline.messages import CompletedEvent
from aidetector.utils.config import DiskConfig


class DiskSink:
    def __init__(self, config: DiskConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.directory = (
            Path("detections") / config.directory if config.directory else None
        )
        if self.directory:
            self.directory.mkdir(parents=True, exist_ok=True)

    def send(self, message: CompletedEvent) -> None:
        event = message.event
        self.logger.info("Saving %d frames to disk", len(event.observations))
        timestamp = date_path(event.best, "seconds")
        label = max(
            event.best.confidences.items(),
            key=lambda item: item[1],
            default=("unclassified", 0),
        )[0]
        directory = self.directory or Path("detections") / label
        target = directory / message.status / f"{timestamp}_{event.event_id}"
        target.mkdir(parents=True, exist_ok=True)

        if self.config.strategy == "ALL":
            for observation in event.observations:
                (target / timestamped_filename(observation)).write_bytes(
                    frame_jpg(observation.frame)
                )

        plot = message.artifacts.image(plot=True)
        clean = message.artifacts.image()
        if plot is not None:
            (target / "best.jpg").write_bytes(plot)
        if clean is not None:
            (target / "clean.jpg").write_bytes(clean)
        video = message.artifacts.video(padding=self.config.crop_padding)
        if video is not None:
            (target / "video.mp4").write_bytes(video)

        metadata = DetectionMetadata.from_event(event, message.validated)
        (target / "metadata.json").write_text(json.dumps(metadata.as_dict()) + "\n")
