from dataclasses import replace

from aidetector.domain.detections import Observation
from aidetector.domain.frames import Frame
from aidetector.media.rendering import get_image

STORED_FRAME_QUALITY = 90


def compact_observation(observation: Observation) -> Observation:
    frame = observation.frame
    if frame.jpeg is not None:
        return observation
    return replace(
        observation,
        frame=Frame(
            frame.captured_at,
            jpeg=get_image(frame.require_image(), STORED_FRAME_QUALITY),
        ),
    )
