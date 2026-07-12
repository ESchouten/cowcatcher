import logging
from concurrent.futures import Executor, ThreadPoolExecutor
from datetime import datetime, timedelta
from threading import Lock

from aidetector.detection.events import DetectionEvent
from aidetector.detection.models import matching_confidences
from aidetector.detection.validator import Validator
from aidetector.exporters.exporter import Exporter
from aidetector.utils.config import YoloConfig

logger = logging.getLogger(__name__)


class ExportDispatcher:
    """Serializes validation and exports without blocking model inference."""

    def __init__(
        self,
        validator: Validator,
        exporters: list[Exporter],
        model_config: YoloConfig | None,
        *,
        executor: Executor | None = None,
    ):
        self.validator = validator
        self.exporters = list(exporters)
        self.model_config = model_config
        self._executor = executor or ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="detection-export",
        )
        self._owns_executor = executor is None
        self._last_detection: dict[str, dict[str, datetime]] = {}
        self._closed = False
        self._lock = Lock()

    def submit(self, event: DetectionEvent) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("Export dispatcher is closed")
            self._executor.submit(self._dispatch, event)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        if self._owns_executor:
            self._executor.shutdown(wait=True)
        for exporter in self.exporters:
            try:
                exporter.close()
            except Exception:
                logger.exception("Failed to close %s", exporter.__class__.__name__)

    def _dispatch(self, event: DetectionEvent) -> None:
        matching = (
            matching_confidences(event.best.confidence, self.model_config.confidence)
            if self.model_config
            else []
        )
        if not self._cooldown_exceeded(event.source, matching, event.best.date):
            logger.info("Skipping %s during cooldown for %s", event.source, matching)
            return

        try:
            validated = self.validator.validate(event)
        except Exception:
            logger.exception("Detection validation failed")
            return

        if validated is not False:
            self._record_detection(event.source, matching, event.best.date)

        for exporter in self.exporters:
            try:
                exporter.export(event, validated)
            except Exception:
                logger.exception("Exporter %s failed", exporter.__class__.__name__)

    def _cooldown_exceeded(
        self,
        source: str,
        classes: list[str],
        detected_at: datetime,
    ) -> bool:
        if self.model_config is None:
            return True
        if not classes:
            return False
        previous = self._last_detection.get(source, {})
        return any(
            detected_at - previous.get(class_name, datetime.min)
            > timedelta(seconds=self._cooldown_for(class_name))
            for class_name in classes
        )

    def _cooldown_for(self, class_name: str) -> float:
        assert self.model_config is not None
        cooldown = self.model_config.cooldown
        if isinstance(cooldown, dict):
            return cooldown.get(class_name, 0)
        return cooldown

    def _record_detection(
        self,
        source: str,
        classes: list[str],
        detected_at: datetime,
    ) -> None:
        previous = self._last_detection.setdefault(source, {})
        for class_name in classes:
            previous[class_name] = detected_at
