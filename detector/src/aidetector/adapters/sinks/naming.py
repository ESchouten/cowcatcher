from typing import Literal

from aidetector.domain.detections import Observation, max_confidence


def date_path(
    observation: Observation,
    timespec: Literal["seconds", "milliseconds"],
) -> str:
    return observation.frame.captured_at.isoformat(timespec=timespec).replace(":", "-")


def timestamped_filename(observation: Observation) -> str:
    timestamp = date_path(observation, "milliseconds")
    confidence = round(max_confidence(observation.confidences), 3)
    return f"{timestamp}_{confidence}.jpg"
