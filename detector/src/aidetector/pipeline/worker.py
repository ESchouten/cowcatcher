import logging
from collections.abc import Callable
from queue import Empty, Full, Queue
from threading import Lock, Thread
from typing import Generic, TypeVar

T = TypeVar("T")


class _Stop:
    pass


_STOP = _Stop()


class LatestWinsWorker(Generic[T]):
    def __init__(
        self,
        handler: Callable[[T], None],
        *,
        name: str,
        capacity: int,
        failure_message: str,
    ):
        self.handler = handler
        self.name = name
        self.failure_message = failure_message
        self._queue: Queue[T | _Stop] = Queue(capacity)
        self._thread: Thread | None = None
        self._closed = False
        self._lock = Lock()
        self._logger = logging.getLogger(__name__)

    def start(self) -> None:
        with self._lock:
            if self._thread is not None:
                return
            if self._closed:
                raise RuntimeError(f"{self.name} is closed")
            self._thread = Thread(target=self._run, name=self.name, daemon=True)
            self._thread.start()

    def submit(self, item: T) -> bool:
        with self._lock:
            if self._thread is None or self._closed:
                raise RuntimeError(f"{self.name} is not running")
            return self._put_latest(item)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            thread = self._thread
        if thread is None:
            return
        self._queue.put(_STOP)
        thread.join()

    def _put_latest(self, item: T) -> bool:
        try:
            self._queue.put_nowait(item)
            return False
        except Full:
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except Empty:
                pass
            self._queue.put_nowait(item)
            return True

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if isinstance(item, _Stop):
                    return
                try:
                    self.handler(item)
                except Exception:
                    self._logger.exception(self.failure_message)
            finally:
                self._queue.task_done()
