import logging
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path

import cv2
from imageio_ffmpeg import get_ffmpeg_exe

from aidetector.domain.detections import DetectedObject, Observation, bounding_region
from aidetector.media.rendering import even_width, frame_image, get_crop, get_plot

logger = logging.getLogger(__name__)
FFMPEG_EXIT_TIMEOUT_SECONDS = 30


def generate_mp4(
    observations: Sequence[Observation],
    width: int | None = None,
    crf: int = 0,
    crop: bool = True,
    plot: bool = True,
    data_max: int | None = None,
    padding: float = 0.1,
) -> bytes | None:
    if not observations:
        return None
    try:
        frames = RenderedFrames(observations, crop, plot, padding)
        fps = _frames_per_second(observations)
        return _encode_to_limit(frames, fps, width, crf, data_max)
    except Exception:
        logger.exception("Failed to generate MP4")
        return None


@dataclass(slots=True)
class RenderedFrames:
    observations: Sequence[Observation]
    crop: bool
    plot: bool
    padding: float
    region: DetectedObject | None = field(init=False)
    last_object_index: int = field(init=False, default=-1)

    def __post_init__(self) -> None:
        objects = [
            item for observation in self.observations for item in observation.objects
        ]
        self.region = bounding_region(objects) if self.crop else None
        if self.region is not None:
            self.last_object_index = max(
                index
                for index, observation in enumerate(self.observations)
                if observation.objects
            )

    def __iter__(self):
        last_objects: tuple[DetectedObject, ...] = ()
        for index, observation in enumerate(self.observations):
            last_objects = observation.objects or last_objects
            rendered = (
                get_crop(
                    observation,
                    crop=self.region,
                    plot=self.plot,
                    padding=self.padding,
                    plot_objects=(
                        last_objects if index <= self.last_object_index else ()
                    ),
                )
                if self.region is not None
                else None
            )
            yield (
                rendered
                if rendered is not None
                else (
                    get_plot(observation)
                    if self.plot
                    else frame_image(observation.frame)
                )
            )


def _frames_per_second(observations: Sequence[Observation]) -> float:
    duration = (
        observations[-1].frame.captured_at - observations[0].frame.captured_at
    ).total_seconds()
    return len(observations) / duration if len(observations) > 1 and duration > 0 else 1


def _encode_to_limit(
    frames: RenderedFrames,
    fps: float,
    width: int | None,
    crf: int,
    data_max: int | None,
) -> bytes | None:
    source_width = next(iter(frames)).shape[1]
    base_width = min(width, source_width) if width else source_width
    if data_max is None:
        return _encode_mp4(frames, fps, base_width, crf)

    max_crf = max(35, crf)
    last_video = None
    for target_crf in range(crf, max_crf + 1, 4):
        last_video = _encode_mp4(frames, fps, base_width, target_crf)
        if last_video is None or len(last_video) <= data_max:
            return last_video

    current_width = base_width
    while current_width > 160:
        current_width = _smaller_width(current_width)
        last_video = _encode_mp4(frames, fps, current_width, max_crf)
        if last_video is None or len(last_video) <= data_max:
            return last_video

    if last_video is not None:
        logger.warning(
            "MP4 still exceeds %s bytes at width=%s and crf=%s",
            data_max,
            current_width,
            max_crf,
        )
    return last_video


def _smaller_width(width: int) -> int:
    scaled = max(160, even_width(int(width * 0.85)))
    if scaled >= width:
        return max(160, even_width(width - 2))
    return scaled


def _encode_mp4(
    frames: RenderedFrames,
    fps: float,
    target_width: int,
    crf: int,
) -> bytes | None:
    iterator = iter(frames)
    first = next(iterator)
    height, width = first.shape[:2]
    target_width = even_width(min(target_width, width))
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as output:
        output_path = Path(output.name)

    command = [
        get_ffmpeg_exe(),
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{width}x{height}",
        "-pix_fmt",
        "bgr24",
        "-r",
        str(fps),
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-crf",
        str(crf),
        "-preset",
        "fast",
        "-vf",
        f"scale={target_width}:-2",
        "-pix_fmt",
        "yuv420p",
        "-an",
        str(output_path),
    ]
    process: subprocess.Popen | None = None
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if process.stdin is None:
            logger.error("Failed to open stdin pipe to FFmpeg")
            return None

        for frame in chain((first,), iterator):
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))
            try:
                process.stdin.write(frame.tobytes())
            except BrokenPipeError:
                break

        _, stderr = process.communicate(timeout=FFMPEG_EXIT_TIMEOUT_SECONDS)
        if process.returncode != 0:
            logger.error("FFmpeg error: %s", stderr.decode(errors="replace"))
            return None
        return output_path.read_bytes()
    except subprocess.TimeoutExpired:
        logger.error("FFmpeg did not finish in time")
        return None
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.communicate()
        output_path.unlink(missing_ok=True)
