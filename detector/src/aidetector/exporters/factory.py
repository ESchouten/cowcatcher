from dataclasses import dataclass, replace

from aidetector.exporters.disk import DiskExporter
from aidetector.exporters.exporter import Exporter, TrackPublisher
from aidetector.exporters.sse import SSEExporter, default_sse_endpoint
from aidetector.exporters.telegram import TelegramExporter
from aidetector.exporters.webhook import WebhookExporter
from aidetector.utils.config import ExportersConfig, config_list


@dataclass(frozen=True)
class ExportTargets:
    exporters: list[Exporter]
    track_publishers: list[TrackPublisher]


def build_exporters(
    config: ExportersConfig | None,
    detector_index: int,
) -> ExportTargets:
    if config is None:
        return ExportTargets([], [])

    telegram = [TelegramExporter(item) for item in config_list(config.telegram)]
    webhooks = [WebhookExporter(item) for item in config_list(config.webhook)]
    disks = [DiskExporter(item) for item in config_list(config.disk)]
    streams: list[SSEExporter] = []
    try:
        for item in config_list(config.sse):
            streams.append(
                SSEExporter(
                    replace(
                        item,
                        endpoint=item.endpoint or default_sse_endpoint(detector_index),
                    )
                )
            )
    except Exception:
        for stream in streams:
            stream.close()
        raise
    return ExportTargets(
        [*telegram, *webhooks, *disks, *streams],
        list(streams),
    )
