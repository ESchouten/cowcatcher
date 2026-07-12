from threading import Lock
from typing import Any

from aidetector.adapters.sinks.metadata import CropMetadata, DetectionMetadata
from aidetector.adapters.sinks.sse_transport import SSEHub, SSEServer
from aidetector.domain.detections import DetectedObject
from aidetector.domain.events import LiveObservation
from aidetector.pipeline.messages import CompletedEvent
from aidetector.utils.config import SSEConfig


class SSESink:
    _servers: dict[int, SSEServer] = {}
    _servers_lock = Lock()

    def __init__(self, config: SSEConfig):
        self.config = config
        self.endpoint = _normalize_endpoint(config.endpoint)
        self.server: SSEServer | None = None
        self.hub: SSEHub | None = None
        self._closed = False

    def start(self) -> None:
        with self._servers_lock:
            if self.server is not None:
                return
            if self._closed:
                raise RuntimeError("SSE sink is closed")
            self.server = self._servers.get(self.config.port) or SSEServer(
                self.config.port
            )
            self._servers[self.config.port] = self.server
            self.server.references += 1
            self.hub = self.server.hub(self.endpoint)

    def send(self, message: LiveObservation | CompletedEvent) -> None:
        if self.hub is None:
            raise RuntimeError("SSE sink is not running")
        if isinstance(message, LiveObservation):
            self.hub.publish("tracks", tracks_payload(message))
            return
        payload = DetectionMetadata.from_event(
            message.event,
            message.validated,
        ).as_dict()
        self.hub.publish("detection", {"type": "detection", **payload})

    def close(self) -> None:
        with self._servers_lock:
            if self._closed:
                return
            self._closed = True
            server = self.server
            self.server = None
            self.hub = None
            if server is None:
                return
            server.references -= 1
            if server.references:
                return
            self._servers.pop(self.config.port, None)
        server.close()


def default_sse_endpoint(pipeline_index: int) -> str:
    return f"/events/{pipeline_index}"


def tracks_payload(message: LiveObservation) -> dict[str, Any]:
    observation = message.observation
    height, width = observation.frame.require_image().shape[:2]
    return {
        "type": "tracks",
        "source": message.source,
        "timestamp": observation.frame.captured_at.isoformat(),
        "width": width,
        "height": height,
        "objects": [
            _track_payload(item, index)
            for index, item in enumerate(observation.objects)
        ],
    }


def _track_payload(item: DetectedObject, index: int) -> dict[str, Any]:
    return {
        "id": item.track_id if item.track_id is not None else index,
        "track_id": item.track_id,
        "crop": CropMetadata.from_object(item).as_dict(),
    }


def _normalize_endpoint(endpoint: str | None) -> str:
    endpoint = (endpoint or "").strip() or default_sse_endpoint(0)
    return endpoint if endpoint.startswith("/") else f"/{endpoint}"
