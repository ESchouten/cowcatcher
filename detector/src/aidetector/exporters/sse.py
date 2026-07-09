import json
import logging
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from queue import Empty, Full, Queue
from threading import Lock, Thread
from typing import Any

from aidetector.detection.tracks import tracks_payload
from aidetector.exporters.exporter import Exporter
from aidetector.utils.config import Detection, SSEConfig, max_confidence

logger = logging.getLogger(__name__)
SSE_HOST = "0.0.0.0"
SSE_KEEPALIVE_SECONDS = 15


class SSEExporter(Exporter[SSEConfig]):
    _servers: dict[int, "_SSEServer"] = {}
    _servers_lock = Lock()

    def __init__(self, config: SSEConfig):
        super().__init__(config)
        with self._servers_lock:
            server = self._servers.get(config.port)
            if server is None:
                server = _SSEServer(config.port)
                self._servers[config.port] = server
            self.hub = server.hub(config.endpoint)

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


class _SSEServer:
    def __init__(self, port: int):
        self.port = port
        self.hubs: dict[str, _SSEHub] = {}
        self.hubs_lock = Lock()
        self.server = _ReusableThreadingHTTPServer(
            (SSE_HOST, port),
            self._handler(),
        )
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        logger.info("SSE server listening on http://%s:%s", SSE_HOST, port)

    def hub(self, endpoint: str | None) -> "_SSEHub":
        normalized_endpoint = _normalize_endpoint(endpoint)
        with self.hubs_lock:
            hub = self.hubs.get(normalized_endpoint)
            if hub is None:
                hub = _SSEHub(normalized_endpoint)
                self.hubs[normalized_endpoint] = hub
                logger.info(
                    "SSE endpoint listening on http://%s:%s%s",
                    SSE_HOST,
                    self.port,
                    normalized_endpoint,
                )
            return hub

    def _handler(self):
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                endpoint = self.path.split("?", 1)[0]
                with server.hubs_lock:
                    hub = server.hubs.get(endpoint)
                if hub is None:
                    self.send_response(404)
                    self.end_headers()
                    return

                hub.handle(self)

            def log_message(self, _format, *_args) -> None:
                return

        return Handler


class _SSEHub:
    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        self.clients: list[Queue[str]] = []
        self.clients_lock = Lock()
        self.event_id = 0

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

    def handle(self, handler: BaseHTTPRequestHandler) -> None:
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.send_header("Cache-Control", "no-cache")
        handler.send_header("Connection", "keep-alive")
        handler.send_header("Access-Control-Allow-Origin", "*")
        handler.end_headers()
        client = self.register()
        try:
            handler.wfile.write(b": connected\n\n")
            handler.wfile.flush()
            while True:
                try:
                    message = client.get(timeout=SSE_KEEPALIVE_SECONDS)
                except Empty:
                    message = ": keepalive\n\n"
                handler.wfile.write(message.encode("utf-8"))
                handler.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            self.unregister(client)


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
        "detections": len(detections),
        "start": detections[0].date.isoformat(),
        "end": detections[-1].date.isoformat(),
        "duration": (detections[-1].date - detections[0].date).total_seconds(),
        "crop": asdict(best_detection.images.crop_region)
        if best_detection.images.crop_region
        else None,
        "crops": [asdict(crop) for crop in best_detection.images.crops],
    }


def default_sse_endpoint(detector_index: int) -> str:
    return f"/events/{detector_index}"


def _normalize_endpoint(endpoint: str | None) -> str:
    endpoint = (endpoint or "").strip() or default_sse_endpoint(0)
    return endpoint if endpoint.startswith("/") else f"/{endpoint}"
