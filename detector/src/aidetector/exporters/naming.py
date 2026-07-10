from typing import Literal

from aidetector.detection.models import Detection, max_confidence


def date_path(
    detection: Detection,
    timespec: Literal["seconds", "milliseconds"],
) -> str:
    return detection.date.isoformat(timespec=timespec).replace(":", "-")


def timestamped_filename(detection: Detection) -> str:
    timestamp = date_path(detection, "milliseconds")
    confidence = round(max_confidence(detection.confidence), 3)
    return f"{timestamp}_{confidence}.jpg"
