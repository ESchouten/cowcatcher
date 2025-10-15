from cowcatcher.config import Config, DetectorConfig
from cowcatcher.exporters.disk import DiskExporter
from cowcatcher.exporters.exporter import Exporter
from cowcatcher.exporters.telegram import TelegramExporter


def create_exporters(config: Config, detector_config: DetectorConfig) -> list[Exporter]:
    exporters: list[Exporter] = []
    if detector_config.save_directory is not None:
        exporters.append(DiskExporter(config, detector_config))
    if config.telegram_bot_token is not None and (
        config.telegram_chat_id is not None
        or detector_config.telegram_chat_id is not None
    ):
        exporters.append(TelegramExporter(config, detector_config))
    return exporters
