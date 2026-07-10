from dataclasses import asdict
from typing import Any

from aidetector.utils.config import Crop, Detection


def tracks_payload(source: str, detection: Detection) -> dict[str, Any]:
    height, width = detection.images.jpg.shape[:2]
    return {
        "type": "tracks",
        "source": source,
        "timestamp": detection.date.isoformat(),
        "width": width,
        "height": height,
        "objects": [_object_payload(crop, index) for index, crop in enumerate(detection.images.crops)],
    }


def _object_payload(crop: Crop, index: int) -> dict[str, Any]:
    return {
        "id": index,
        "track_id": crop.track_id,
        "label": crop.label,
        "confidence": crop.confidence,
        "crop": {
            "x1": crop.x1,
            "y1": crop.y1,
            "x2": crop.x2,
            "y2": crop.y2,
            "label": crop.label,
            "confidence": crop.confidence,
        },
        "identity": asdict(crop.identity) if crop.identity else None,
    }
