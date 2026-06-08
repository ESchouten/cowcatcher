import logging
import os
import subprocess
import tempfile

import cv2
import numpy as np
from aidetector.utils.config import Crop, Detection
from imageio_ffmpeg import get_ffmpeg_exe

logger = logging.getLogger(__name__)


def even_width(value: int) -> int:
    return max(2, value // 2 * 2)


def generate_mp4(
    detections: list[Detection],
    width: int | None = None,
    crf: int = 0,
    crop: bool = True,
    plot: bool = True,
    data_max: int | None = None,
    padding: float = 0.1,
) -> bytes | None:
    try:
        if not detections:
            return None

        frames: list[np.ndarray] = []
        if crop:
            crops = [d.images.crop for d in detections if d.images.crop is not None]
            if crops:
                minX1 = min(crop.x1 for crop in crops)
                minY1 = min(crop.y1 for crop in crops)
                maxX2 = max(crop.x2 for crop in crops)
                maxY2 = max(crop.y2 for crop in crops)
                crop_region = Crop(minX1, minY1, maxX2, maxY2)
                frames = [
                    f
                    for d in detections
                    if (f := get_crop(d, crop=crop_region, plot=plot, padding=padding))
                    is not None
                ]

        if not frames:
            frames = [
                d.images.plot if plot and d.images.plot is not None else d.images.jpg
                for d in detections
            ]

        # 1. Calculate FPS
        fps = (
            len(detections) / (detections[-1].date - detections[0].date).total_seconds()
            if len(detections) > 1
            else 1
        )

        # 2. Get dimensions from first frame
        # We need the source dimensions to tell FFmpeg what size the raw input stream is
        h, w = frames[0].shape[:2]

        ffmpeg_exe = get_ffmpeg_exe()

        def encode_mp4(target_width: int, target_crf: int) -> bytes | None:
            # Build the scaling filter string
            target_width = even_width(min(target_width, w))
            vf_scale = f"scale={target_width}:-2"

            # 3. Setup FFmpeg command
            # We write to a unique temp file because MP4 atoms are tricky to stream directly to stdout
            # without using fragmented MP4s (which have lower player compatibility).
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_out:
                output_path = temp_out.name

            try:
                cmd = [
                    ffmpeg_exe,
                    "-y",
                    "-f",
                    "rawvideo",  # Input format is raw pixels
                    "-vcodec",
                    "rawvideo",
                    "-s",
                    f"{w}x{h}",  # Input resolution (Source)
                    "-pix_fmt",
                    "bgr24",  # OpenCV uses BGR, not RGB
                    "-r",
                    str(fps),  # Input Framerate
                    "-i",
                    "-",  # Read from Stdin
                    "-c:v",
                    "libx264",  # Encoder
                    "-crf",
                    str(target_crf),  # Quality
                    "-preset",
                    "fast",  # Encoding speed
                    "-vf",
                    vf_scale,  # Apply scaling here (safer than python)
                    "-pix_fmt",
                    "yuv420p",  # Essential for QuickTime/Web compatibility
                    "-an",  # No audio
                    output_path,
                ]

                # 4. Open Subprocess
                process = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )

                # 5. Feed frames
                if process.stdin is None:
                    logger.error("Failed to open stdin pipe to FFmpeg")
                    return None

                for frame in frames:
                    # Sanity check: ensure frame size matches the stream setup
                    if frame.shape[0] != h or frame.shape[1] != w:
                        frame = cv2.resize(frame, (w, h))

                    try:
                        process.stdin.write(frame.tobytes())
                    except BrokenPipeError:
                        logger.error("FFmpeg process died unexpectedly.")
                        break

                # 6. Finish Encoding
                stdout, stderr = process.communicate()

                if process.returncode != 0:
                    logger.error(f"FFmpeg error: {stderr.decode()}")
                    return None

                # 7. Read bytes
                with open(output_path, "rb") as f:
                    video_bytes = f.read()

                return video_bytes

            finally:
                if os.path.exists(output_path):
                    os.remove(output_path)

        base_width = min(width, w) if width else w
        if data_max is None:
            return encode_mp4(base_width, crf)

        max_crf = max(35, crf)
        crf_step = 4
        min_width = 160
        width_step = 0.85

        last_video = None

        for target_crf in range(crf, max_crf + 1, crf_step):
            last_video = encode_mp4(base_width, target_crf)
            if last_video is None:
                return None
            if len(last_video) <= data_max:
                return last_video

        current_width = base_width
        while current_width > min_width:
            next_width = int(current_width * width_step)
            next_width = max(min_width, even_width(next_width))
            if next_width >= current_width:
                next_width = max(min_width, even_width(current_width - 2))
            if next_width == current_width:
                break
            current_width = next_width
            last_video = encode_mp4(current_width, max_crf)
            if last_video is None:
                return None
            if len(last_video) <= data_max:
                return last_video

        if last_video is not None:
            logger.warning(
                "MP4 still exceeds %s bytes at width=%s and crf=%s",
                data_max,
                current_width,
                max_crf,
            )
        return last_video

    except Exception:
        logger.exception("Failed to generate MP4")
        return None


def get_image(image: np.ndarray, quality: int = 100) -> bytes:
    success, jpg = cv2.imencode(".jpg", image, (int(cv2.IMWRITE_JPEG_QUALITY), quality))
    if not success:
        raise ValueError("Failed to encode image")
    return jpg.tobytes()


def shrink_image(image: np.ndarray, width_max: int) -> np.ndarray:
    h, w = image.shape[:2]
    width_max = even_width(width_max)
    if w <= width_max:
        return image

    scale = width_max / w
    new_h = even_width(round(h * scale))
    return cv2.resize(image, (width_max, new_h), interpolation=cv2.INTER_AREA)


def compress_jpg(
    image: np.ndarray,
    max_bytes: int,
    start_quality: int = 90,
    min_quality: int = 10,
    min_scale: float = 0.1,
    quality_step: int = 10,
    scale_step: float = 0.9,
) -> bytes | None:
    img = image
    quality = start_quality
    jpg = get_image(img, quality)

    while len(jpg) > max_bytes and quality > min_quality:
        quality = max(min_quality, quality - quality_step)
        jpg = get_image(img, quality)

    scale = 1.0
    while len(jpg) > max_bytes and scale > min_scale:
        scale = max(min_scale, scale * scale_step)
        width = max(1, int(img.shape[1] * scale))
        height = max(1, int(img.shape[0] * scale))
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        jpg = get_image(img, quality)

    return jpg


def get_crop(
    detection: Detection,
    crop: Crop | None = None,
    aspect_ratio: float | None = 16 / 9,
    padding: float = 0.1,
    plot: bool = True,
) -> np.ndarray | None:
    def centered_range(center: float, size: int, limit: int) -> tuple[int, int]:
        size = max(1, min(size, limit))
        start = int(round(center - size / 2))
        end = start + size

        if start < 0:
            end -= start
            start = 0
        if end > limit:
            start -= end - limit
            end = limit

        return max(0, start), min(limit, end)

    crop = crop or detection.images.crop
    if crop is None:
        return None
    img = (
        detection.images.plot
        if plot and detection.images.plot is not None
        else detection.images.jpg
    )
    h, w = img.shape[:2]
    box_w, box_h = (
        max(1, crop.x2 - crop.x1),
        max(1, crop.y2 - crop.y1),
    )
    pad_x, pad_y = int(box_w * padding), int(box_h * padding)
    x1, y1 = max(0, crop.x1 - pad_x), max(0, crop.y1 - pad_y)
    x2, y2 = min(w, crop.x2 + pad_x), min(h, crop.y2 + pad_y)

    if x2 <= x1 or y2 <= y1:
        return None

    if aspect_ratio and aspect_ratio > 0:
        crop_w = x2 - x1
        crop_h = y2 - y1
        target_w = crop_w
        target_h = crop_h

        current_ratio = crop_w / crop_h
        if current_ratio < aspect_ratio:
            target_w = int(round(crop_h * aspect_ratio))
        elif current_ratio > aspect_ratio:
            target_h = int(round(crop_w / aspect_ratio))

        if target_w > w:
            target_w = w
            target_h = int(round(target_w / aspect_ratio))
        if target_h > h:
            target_h = h
            target_w = int(round(target_h * aspect_ratio))

        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        x1, x2 = centered_range(center_x, target_w, w)
        y1, y2 = centered_range(center_y, target_h, h)

    return img[y1:y2, x1:x2]
