import json
import logging
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from queue import Empty, Full, Queue
from threading import Lock, Thread
from typing import Any

from aidetector.detection.models import Detection, max_confidence
from aidetector.detection.tracks import crop_payload, tracks_payload
from aidetector.exporters.exporter import Exporter
from aidetector.utils.config import SSEConfig

logger = logging.getLogger(__name__)
SSE_HOST = os.environ.get("SSE_HOST", "127.0.0.1")
SSE_KEEPALIVE_SECONDS = 15


class SSEExporter(Exporter[SSEConfig]):
    _servers: dict[int, "_SSEServer"] = {}
    _servers_lock = Lock()

    def __init__(self, config: SSEConfig):
        super().__init__(config)
        with self._servers_lock:
            self.server = self._servers.get(config.port) or _SSEServer(config.port)
            self._servers[config.port] = self.server
            self.server.references += 1
        self.hub = self.server.hub(config.endpoint)
        self._closed = False

    def publish_tracks(self, source: str, detection: Detection) -> None:
        self.hub.publish("tracks", tracks_payload(source, detection))

    def _export(
        self,
        best_detection: Detection,
        detections: list[Detection],
        validated: bool | None,
    ) -> None:
        self.hub.publish(
            "detection",
            detection_payload(best_detection, detections, validated),
        )

    def close(self) -> None:
        with self._servers_lock:
            if self._closed:
                return
            self._closed = True
            self.server.references -= 1
            if self.server.references:
                return
            self._servers.pop(self.config.port, None)
        self.server.close()


class _ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True
    block_on_close = False


class _SSEServer:
    def __init__(self, port: int):
        self.port = port
        self.references = 0
        self.hubs: dict[str, _SSEHub] = {}
        self.hubs_lock = Lock()
        self.server = _ReusableThreadingHTTPServer(
            (SSE_HOST, port),
            self._handler(),
        )
        self.thread = Thread(
            target=self.server.serve_forever,
            name=f"sse-{port}",
            daemon=True,
        )
        self.thread.start()
        logger.info("SSE server listening on http://%s:%s", SSE_HOST, port)

    def hub(self, endpoint: str | None) -> "_SSEHub":
        normalized = _normalize_endpoint(endpoint)
        with self.hubs_lock:
            hub = self.hubs.get(normalized)
            if hub is None:
                hub = _SSEHub(normalized)
                self.hubs[normalized] = hub
            return hub

    def close(self) -> None:
        with self.hubs_lock:
            hubs = list(self.hubs.values())
            self.hubs.clear()
        for hub in hubs:
            hub.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def _handler(self):
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                endpoint = self.path.split("?", 1)[0]
                with owner.hubs_lock:
                    hub = owner.hubs.get(endpoint)
                if hub is None:
                    self.send_error(404)
                    return
                hub.handle(self)

            def log_message(self, format: str, *args: Any) -> None:
                logger.debug(format, *args)

        return Handler


class _SSEHub:
    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        self.clients: list[Queue[str | None]] = []
        self.lock = Lock()
        self.event_id = 0
        self.closed = False

    def publish(self, event: str, data: dict[str, Any]) -> None:
        with self.lock:
            if self.closed:
                return
            self.event_id += 1
            message = _format_event(self.event_id, event, data)
            clients = list(self.clients)
        for client in clients:
            try:
                client.put_nowait(message)
            except Full:
                _replace_queued_message(client, message)

    def register(self) -> Queue[str | None]:
        client: Queue[str | None] = Queue(maxsize=100)
        with self.lock:
            if self.closed:
                raise RuntimeError("SSE endpoint is closed")
            self.clients.append(client)
        return client

    def unregister(self, client: Queue[str | None]) -> None:
        with self.lock:
            if client in self.clients:
                self.clients.remove(client)

    def close(self) -> None:
        with self.lock:
            self.closed = True
            clients = list(self.clients)
            self.clients.clear()
        for client in clients:
            _close_queue(client)

    def handle(self, handler: BaseHTTPRequestHandler) -> None:
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.send_header("Cache-Control", "no-cache")
        handler.send_header("Connection", "keep-alive")
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
                if message is None:
                    return
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
    region = best_detection.images.crop_region
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
        "crop": crop_payload(region) if region else None,
        "crops": [crop_payload(crop) for crop in best_detection.images.crops],
    }


def default_sse_endpoint(detector_index: int) -> str:
    return f"/events/{detector_index}"


def _normalize_endpoint(endpoint: str | None) -> str:
    endpoint = (endpoint or "").strip() or default_sse_endpoint(0)
    return endpoint if endpoint.startswith("/") else f"/{endpoint}"


def _format_event(event_id: int, event: str, data: dict[str, Any]) -> str:
    payload = json.dumps(data, separators=(",", ":"))
    return f"id: {event_id}\nevent: {event}\ndata: {payload}\n\n"


def _replace_queued_message(queue: Queue[str | None], message: str) -> None:
    try:
        queue.get_nowait()
    except Empty:
        pass
    try:
        queue.put_nowait(message)
    except Full:
        pass


def _close_queue(queue: Queue[str | None]) -> None:
    while True:
        try:
            queue.get_nowait()
        except Empty:
            break
    queue.put_nowait(None)
