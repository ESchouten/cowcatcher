import logging
import os
import subprocess
import tempfile

import cv2
import numpy as np
from aidetector.detection.models import Crop, Detection
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
            crops = [crop for d in detections for crop in d.images.crops]
            if crops:
                crop_region = Crop(
                    min(item.x1 for item in crops),
                    min(item.y1 for item in crops),
                    max(item.x2 for item in crops),
                    max(item.y2 for item in crops),
                )
                last_crop_index = max(
                    i for i, d in enumerate(detections) if d.images.crops
                )
                last_crops: list[Crop] = []
                for i, detection in enumerate(detections):
                    last_crops = detection.images.crops or last_crops
                    frame = get_crop(
                        detection,
                        crop=crop_region,
                        plot=plot,
                        padding=padding,
                        plot_crops=last_crops if i <= last_crop_index else [],
                    )
                    if frame is not None:
                        frames.append(frame)

        if not frames:
            frames = [get_plot(d) if plot else d.images.jpg for d in detections]

        duration = (detections[-1].date - detections[0].date).total_seconds()
        fps = len(detections) / duration if len(detections) > 1 and duration > 0 else 1

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
                    "-loglevel",
                    "error",
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
    success, jpg = cv2.imencode(
        ".jpg",
        image,
        (int(cv2.IMWRITE_JPEG_QUALITY), quality),
    )
    if not success:
        raise ValueError("Failed to encode image")
    return jpg.tobytes()


def get_plot(detection: Detection, crops: list[Crop] | None = None) -> np.ndarray:
    crops = crops if crops is not None else detection.images.crops
    if not crops:
        return detection.images.jpg

    image = detection.images.jpg.copy()
    h, w = image.shape[:2]
    color = (255, 0, 0)
    thickness = max(2, round(min(w, h) / 500))
    font_scale = max(0.5, min(w, h) / 1200)
    font_thickness = max(1, round(thickness / 2))

    for crop in crops:
        x1 = max(0, min(w - 1, crop.x1))
        y1 = max(0, min(h - 1, crop.y1))
        x2 = max(0, min(w - 1, crop.x2))
        y2 = max(0, min(h - 1, crop.y2))
        if x2 <= x1 or y2 <= y1:
            continue

        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        if crop.label is None or crop.confidence is None:
            continue

        label = f"{crop.label} {crop.confidence:.0%}"
        (text_w, text_h), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            font_thickness,
        )
        label_y1 = max(0, y1 - text_h - baseline - thickness)
        label_y2 = label_y1 + text_h + baseline + thickness
        label_x2 = min(w - 1, x1 + text_w + thickness * 2)
        cv2.rectangle(image, (x1, label_y1), (label_x2, label_y2), color, -1)
        cv2.putText(
            image,
            label,
            (x1 + thickness, label_y2 - baseline - max(1, thickness // 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            font_thickness,
            cv2.LINE_AA,
        )
    return image


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
) -> bytes:
    quality = start_quality
    jpg = get_image(image, quality)

    while len(jpg) > max_bytes and quality > min_quality:
        quality = max(min_quality, quality - quality_step)
        jpg = get_image(image, quality)

    scale = 1.0
    while len(jpg) > max_bytes and scale > min_scale:
        scale = max(min_scale, scale * scale_step)
        width = max(1, int(image.shape[1] * scale))
        height = max(1, int(image.shape[0] * scale))
        resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        jpg = get_image(resized, quality)

    return jpg


def get_crop(
    detection: Detection,
    crop: Crop | None = None,
    aspect_ratio: float | None = 16 / 9,
    padding: float = 0.1,
    plot: bool = True,
    plot_crops: list[Crop] | None = None,
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

    crop = crop or detection.images.crop_region
    if crop is None:
        return None
    img = get_plot(detection, plot_crops) if plot else detection.images.jpg
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
