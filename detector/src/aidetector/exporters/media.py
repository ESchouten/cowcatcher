from dataclasses import dataclass

from aidetector.detection.events import DetectionEvent
from aidetector.exporters.naming import timestamped_filename
from aidetector.media.video import (
    compress_jpg,
    generate_mp4,
    get_crop,
    get_image,
    get_plot,
)
from aidetector.utils.config import MediaExporterConfig


@dataclass(frozen=True)
class EncodedFile:
    filename: str
    content: bytes
    content_type: str

    def request_file(self) -> tuple[str, bytes, str]:
        return self.filename, self.content, self.content_type


def encode_media(
    event: DetectionEvent,
    config: MediaExporterConfig,
    *,
    data_max: int | None = None,
) -> dict[str, EncodedFile]:
    detection = event.best
    filename = timestamped_filename(detection)
    media: dict[str, EncodedFile] = {}
    if config.include_image:
        media["image"] = EncodedFile(
            filename,
            _encode_jpg(detection.images.jpg, data_max),
            "image/jpeg",
        )
    if config.include_plot:
        media["photo"] = EncodedFile(
            filename,
            _encode_jpg(get_plot(detection), data_max),
            "image/jpeg",
        )
    if config.include_crop:
        crop = get_crop(detection, padding=config.crop_padding)
        if crop is not None:
            media["crop"] = EncodedFile(
                filename.replace(".jpg", "_crop.jpg"),
                _encode_jpg(crop, data_max),
                "image/jpeg",
            )
    if config.include_video:
        video = generate_mp4(
            event.detections,
            width=config.video_width,
            crf=config.video_crf,
            data_max=data_max,
            padding=config.crop_padding,
        )
        if video is not None:
            media["video"] = EncodedFile(
                filename.replace(".jpg", ".mp4"),
                video,
                "video/mp4",
            )
    return media


def _encode_jpg(image, data_max: int | None) -> bytes:
    if data_max is None:
        return get_image(image)
    return compress_jpg(image, data_max)
