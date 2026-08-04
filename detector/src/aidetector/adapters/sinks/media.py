from dataclasses import dataclass

from aidetector.adapters.sinks.naming import timestamped_filename
from aidetector.pipeline.messages import CompletedEvent
from aidetector.utils.config import MediaExporterConfig


@dataclass(frozen=True, slots=True)
class EncodedFile:
    filename: str
    content: bytes
    content_type: str

    def request_file(self) -> tuple[str, bytes, str]:
        return self.filename, self.content, self.content_type


def encode_media(
    message: CompletedEvent,
    config: MediaExporterConfig,
    *,
    data_max: int | None = None,
) -> dict[str, EncodedFile]:
    filename = timestamped_filename(message.event.best)
    media: dict[str, EncodedFile] = {}
    if config.include_image:
        _add_image(
            media,
            "image",
            filename,
            message.artifacts.image(data_max=data_max),
        )
    if config.include_plot:
        _add_image(
            media,
            "photo",
            filename,
            message.artifacts.image(plot=True, data_max=data_max),
        )
    if config.include_crop:
        _add_image(
            media,
            "crop",
            filename.replace(".jpg", "_crop.jpg"),
            message.artifacts.image(
                crop=True,
                plot=True,
                padding=config.crop_padding,
                data_max=data_max,
            ),
        )
    if config.include_video:
        video = message.artifacts.video(
            width=config.video_width,
            crf=config.video_crf,
            padding=config.crop_padding,
            data_max=data_max,
        )
        if video is not None:
            media["video"] = EncodedFile(
                filename.replace(".jpg", ".mp4"), video, "video/mp4"
            )
    return media


def _add_image(
    media: dict[str, EncodedFile],
    name: str,
    filename: str,
    content: bytes | None,
) -> None:
    if content is not None:
        media[name] = EncodedFile(filename, content, "image/jpeg")
