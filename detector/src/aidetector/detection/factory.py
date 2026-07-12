from aidetector.detection.detector import Detector, ModelRuntime
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
    model = (
        ModelRuntime(
            config.yolo,
            YoloRunner(config.yolo, onnx, source_provider.sources),
            EventCollector(config.yolo),
        )
        if config.yolo
        else None
    )
    vlms = [config.vlm] if isinstance(config.vlm, VLMConfig) else list(config.vlm or [])
    validator = Validator(vlms)
    targets = build_exporters(config.exporters, detector_index)
    dispatcher = ExportDispatcher(validator, targets.exporters, config.yolo)
    return Detector(
        config.detection,
        source_provider,
        model,
        dispatcher,
        targets.track_publishers,
        detector_index,
    )
