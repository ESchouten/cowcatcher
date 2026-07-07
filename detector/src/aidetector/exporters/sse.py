import json
import logging
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Lock, Thread
from typing import Any

from aidetector.exporters.exporter import Exporter
from aidetector.utils.config import (
    DetectedObject,
    Detection,
    IdentityResult,
    SSEConfig,
    max_confidence,
)

logger = logging.getLogger(__name__)
SSE_HOST = "0.0.0.0"
tracks_log_lock = Lock()


class SSEExporter(Exporter[SSEConfig]):
    _servers: dict[int, "_SSEHub"] = {}
    _servers_lock = Lock()

    def __init__(self, config: SSEConfig):
        super().__init__(config)
        with self._servers_lock:
            key = config.port
            hub = self._servers.get(key)
            if hub is None:
                hub = _SSEHub(config)
                self._servers[key] = hub
            elif hub.endpoint != config.endpoint:
                raise ValueError(
                    f"SSE server already running on {SSE_HOST}:{config.port} "
                    f"with endpoint {hub.endpoint}"
                )
        self.hub = hub

    def publish_tracks(self, source: str, detection: Detection) -> None:
        self.hub.publish("tracks", tracks_payload(source, detection))

    def filtered_export(
        self,
        best_detection: Detection,
        detections: list[Detection],
        validated: bool | None,
    ) -> None:
        self.hub.publish(
            "detection",
            detection_payload(best_detection, detections, validated),
        )


class _ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


class _SSEHub:
    def __init__(self, config: SSEConfig):
        self.endpoint = _normalize_endpoint(config.endpoint)
        self.keepalive = max(1, config.keepalive)
        self.clients: list[Queue[str]] = []
        self.clients_lock = Lock()
        self.event_id = 0
        self.server = _ReusableThreadingHTTPServer(
            (SSE_HOST, config.port),
            self._handler(),
        )
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        logger.info(
            "SSE exporter listening on http://%s:%s%s",
            SSE_HOST,
            config.port,
            self.endpoint,
        )

    def publish(self, event: str, data: dict[str, Any]) -> None:
        message = self._format_event(event, data)
        with self.clients_lock:
            clients = list(self.clients)
        for client in clients:
            try:
                client.put_nowait(message)
            except Full:
                try:
                    client.get_nowait()
                except Empty:
                    pass
                try:
                    client.put_nowait(message)
                except Full:
                    pass

    def register(self) -> Queue[str]:
        client: Queue[str] = Queue(maxsize=100)
        with self.clients_lock:
            self.clients.append(client)
        return client

    def unregister(self, client: Queue[str]) -> None:
        with self.clients_lock:
            if client in self.clients:
                self.clients.remove(client)

    def _format_event(self, event: str, data: dict[str, Any]) -> str:
        self.event_id += 1
        payload = json.dumps(data, separators=(",", ":"))
        return f"id: {self.event_id}\nevent: {event}\ndata: {payload}\n\n"

    def _handler(self):
        hub = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if self.path.split("?", 1)[0] != hub.endpoint:
                    self.send_response(404)
                    self.end_headers()
                    return

                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                client = hub.register()
                try:
                    self.wfile.write(b": connected\n\n")
                    self.wfile.flush()
                    while True:
                        try:
                            message = client.get(timeout=hub.keepalive)
                        except Empty:
                            message = ": keepalive\n\n"
                        self.wfile.write(message.encode("utf-8"))
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                finally:
                    hub.unregister(client)

            def log_message(self, _format, *_args) -> None:
                return

        return Handler


def tracks_payload(source: str, detection: Detection) -> dict[str, Any]:
    height, width = detection.images.jpg.shape[:2]
    return {
        "type": "tracks",
        "source": source,
        "timestamp": detection.date.isoformat(),
        "width": width,
        "height": height,
        "objects": [_object_payload(obj) for obj in detection.images.objects],
    }


def detection_payload(
    best_detection: Detection,
    detections: list[Detection],
    validated: bool | None,
) -> dict[str, Any]:
    return {
        "type": "detection",
        "timestamp": best_detection.date.isoformat(),
        "confidence": max_confidence(best_detection.confidence),
        "confidences": best_detection.confidence,
        "validated": validated,
        "duration": (detections[-1].date - detections[0].date).total_seconds(),
        "identity": _identity_payload(best_detection.identities[0])
        if best_detection.identities
        else None,
        "identities": [
            _identity_payload(identity) for identity in best_detection.identities
        ],
    }


def _object_payload(obj: DetectedObject) -> dict[str, Any]:
    crop = obj.crop
    return {
        "track_id": obj.track_id,
        "label": crop.label,
        "confidence": crop.confidence,
        "crop": {
            "x1": crop.x1,
            "y1": crop.y1,
            "x2": crop.x2,
            "y2": crop.y2,
        },
        "identity": _identity_payload(obj.identity) if obj.identity else None,
    }


def _identity_payload(identity: IdentityResult) -> dict[str, Any]:
    return asdict(identity)


def write_tracks_log(log_file: Path, payload: dict[str, Any]) -> None:
    try:
        with tracks_log_lock:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            with log_file.open("a", encoding="utf-8") as file:
                json.dump(payload, file, separators=(",", ":"))
                file.write("\n")
    except Exception:
        logger.exception("Failed to write identity SSE log")


def _normalize_endpoint(endpoint: str) -> str:
    endpoint = endpoint.strip() or "/events"
    return endpoint if endpoint.startswith("/") else f"/{endpoint}"
