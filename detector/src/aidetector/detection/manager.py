from time import sleep

from aidetector.detection.detector import Detector
from aidetector.detection.factory import build_detector
from aidetector.services.healthcheck import Healthcheck
from aidetector.utils.config import Config


class Manager:
    def __init__(self, detectors: list[Detector], health: Healthcheck | None):
        self.detectors = detectors
        self.health = health

    @classmethod
    def from_config(cls, config: Config) -> "Manager":
        detectors: list[Detector] = []
        try:
            for index, detector_config in enumerate(config.detectors):
                detectors.append(build_detector(detector_config, config.onnx, index))
        except Exception:
            for detector in detectors:
                detector.close()
            raise
        health = Healthcheck(config.health) if config.health else None
        return cls(detectors, health)

    def start(self) -> None:
        try:
            for detector in self.detectors:
                detector.start()
            if self.health:
                self.health.start()
        except Exception:
            self.stop()
            raise

    def wait(self) -> None:
        while True:
            failed = next(
                (detector for detector in self.detectors if detector.error is not None),
                None,
            )
            if failed is not None:
                raise RuntimeError("Detector stopped unexpectedly") from failed.error
            if not any(detector.is_alive for detector in self.detectors):
                return
            sleep(0.1)

    def stop(self) -> None:
        for detector in self.detectors:
            detector.stop()
        if self.health:
            self.health.stop()
        for detector in self.detectors:
            detector.close()
