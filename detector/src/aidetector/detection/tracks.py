from typing import Any

from aidetector.detection.models import Crop, Detection


def tracks_payload(source: str, detection: Detection) -> dict[str, Any]:
    height, width = detection.images.jpg.shape[:2]
    return {
        "type": "tracks",
        "source": source,
        "timestamp": detection.date.isoformat(),
        "width": width,
        "height": height,
        "objects": [
            _object_payload(crop, index)
            for index, crop in enumerate(detection.images.crops)
        ],
    }


def _object_payload(crop: Crop, index: int) -> dict[str, Any]:
    return {
        "id": crop.track_id if crop.track_id is not None else index,
        "track_id": crop.track_id,
        "label": crop.label,
        "confidence": crop.confidence,
        "crop": crop_payload(crop),
    }


def crop_payload(crop: Crop) -> dict[str, Any]:
    return {
        "x1": crop.x1,
        "y1": crop.y1,
        "x2": crop.x2,
        "y2": crop.y2,
        "label": crop.label,
        "confidence": crop.confidence,
    }
