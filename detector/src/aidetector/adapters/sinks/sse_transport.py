import json
import logging
import os
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Condition, Lock, Thread
from typing import Any

logger = logging.getLogger(__name__)
SSE_HOST = os.environ.get("SSE_HOST", "127.0.0.1")
SSE_KEEPALIVE_SECONDS = 15


class _ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True
    block_on_close = False


class SSEServer:
    def __init__(self, port: int):
        self.port = port
        self.references = 0
        self.hubs: dict[str, SSEHub] = {}
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

    def hub(self, endpoint: str) -> "SSEHub":
        with self.hubs_lock:
            hub = self.hubs.get(endpoint)
            if hub is None:
                hub = SSEHub(endpoint)
                self.hubs[endpoint] = hub
            return hub

    def close(self) -> None:
        with self.hubs_lock:
            hubs = list(self.hubs.values())
            self.hubs.clear()
        for hub in hubs:
            hub.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

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


class SSEHub:
    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        self.clients: list[SSEClient] = []
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
            client.publish(event, message)

    def register(self) -> "SSEClient":
        client = SSEClient()
        with self.lock:
            if self.closed:
                raise RuntimeError("SSE endpoint is closed")
            self.clients.append(client)
        return client

    def unregister(self, client: "SSEClient") -> None:
        with self.lock:
            if client in self.clients:
                self.clients.remove(client)

    def close(self) -> None:
        with self.lock:
            self.closed = True
            clients = list(self.clients)
            self.clients.clear()
        for client in clients:
            client.close()

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
                message = client.read(SSE_KEEPALIVE_SECONDS)
                if message is None:
                    return
                handler.wfile.write(message.encode("utf-8"))
                handler.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            self.unregister(client)


class SSEClient:
    def __init__(self, event_capacity: int = 100):
        self.event_capacity = event_capacity
        self.events: deque[str] = deque()
        self.latest_track: str | None = None
        self.closed = False
        self.condition = Condition()

    def publish(self, event: str, message: str) -> None:
        with self.condition:
            if self.closed:
                return
            if event == "tracks":
                self.latest_track = message
            else:
                if len(self.events) == self.event_capacity:
                    self.events.popleft()
                    logger.warning("Dropped completed event for slow SSE client")
                self.events.append(message)
            self.condition.notify()

    def read(self, timeout: float) -> str | None:
        with self.condition:
            ready = self.condition.wait_for(
                lambda: (
                    self.closed or bool(self.events) or self.latest_track is not None
                ),
                timeout,
            )
            if not ready:
                return ": keepalive\n\n"
            if self.events:
                return self.events.popleft()
            if self.latest_track is not None:
                message = self.latest_track
                self.latest_track = None
                return message
            return None

    def close(self) -> None:
        with self.condition:
            self.closed = True
            self.events.clear()
            self.latest_track = None
            self.condition.notify_all()


def _format_event(event_id: int, event: str, data: dict[str, Any]) -> str:
    payload = json.dumps(data, separators=(",", ":"))
    return f"id: {event_id}\nevent: {event}\ndata: {payload}\n\n"
