from dataclasses import dataclass

from aidetector.detection.models import Detection
from aidetector.exporters.naming import timestamped_filename
from aidetector.media.video import (
    compress_jpg,
    generate_mp4,
    get_crop,
    get_image,
    get_plot,
)


@dataclass(frozen=True)
class MediaOptions:
    include_image: bool
    include_plot: bool
    include_crop: bool
    include_video: bool
    video_width: int | None
    video_crf: int
    crop_padding: float
    data_max: int | None = None


@dataclass(frozen=True)
class EncodedFile:
    filename: str
    content: bytes
    content_type: str

    def request_file(self) -> tuple[str, bytes, str]:
        return self.filename, self.content, self.content_type


def encode_media(
    detection: Detection,
    detections: list[Detection],
    options: MediaOptions,
) -> dict[str, EncodedFile]:
    filename = timestamped_filename(detection)
    media: dict[str, EncodedFile] = {}
    if options.include_image:
        media["image"] = EncodedFile(
            filename,
            _encode_jpg(detection.images.jpg, options.data_max),
            "image/jpeg",
        )
    if options.include_plot:
        media["photo"] = EncodedFile(
            filename,
            _encode_jpg(get_plot(detection), options.data_max),
            "image/jpeg",
        )
    if options.include_crop:
        crop = get_crop(detection, padding=options.crop_padding)
        if crop is not None:
            media["crop"] = EncodedFile(
                filename.replace(".jpg", "_crop.jpg"),
                _encode_jpg(crop, options.data_max),
                "image/jpeg",
            )
    if options.include_video:
        video = generate_mp4(
            detections,
            width=options.video_width,
            crf=options.video_crf,
            data_max=options.data_max,
            padding=options.crop_padding,
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
