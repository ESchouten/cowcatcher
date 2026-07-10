from aidetector.detection.detector import Detector
from aidetector.detection.events import EventCollector
from aidetector.detection.validator import Validator
from aidetector.detection.yolo import YoloRunner
from aidetector.exporters.dispatcher import ExportDispatcher
from aidetector.exporters.factory import build_exporters
from aidetector.sources.source import SourceProvider
from aidetector.utils.config import DetectorConfig, OnnxConfig, VLMConfig


def build_detector(
    config: DetectorConfig,
    onnx: OnnxConfig,
    detector_index: int,
) -> Detector:
    source_provider = SourceProvider(config.detection)
    runner = (
        YoloRunner(config.yolo, onnx, source_provider.sources)
        if config.yolo is not None
        else None
    )
    collector = EventCollector(config.yolo) if config.yolo is not None else None
    vlms = [config.vlm] if isinstance(config.vlm, VLMConfig) else list(config.vlm or [])
    validator = Validator(vlms)
    targets = build_exporters(config.exporters, detector_index)
    dispatcher = ExportDispatcher(validator, targets.exporters, config.yolo)
    return Detector(
        config.detection,
        config.yolo,
        source_provider,
        runner,
        collector,
        dispatcher,
        targets.track_publishers,
        detector_index,
    )
