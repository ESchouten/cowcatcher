from dataclasses import dataclass, replace

from aidetector.adapters.sinks.disk import DiskSink
from aidetector.adapters.sinks.sse import SSESink, default_sse_endpoint
from aidetector.adapters.sinks.telegram import TelegramSink
from aidetector.adapters.sinks.webhook import WebhookSink
from aidetector.domain.events import LiveObservation
from aidetector.domain.policies import ExportPolicy
from aidetector.pipeline.messages import CompletedEvent
from aidetector.pipeline.ports import Resource, Sink
from aidetector.pipeline.sinks import BufferedSink, FilteredEventSink
from aidetector.utils.config import ExporterConfig, ExportersConfig, config_list


@dataclass(frozen=True, slots=True)
class SinkBundle:
    events: list[Sink[CompletedEvent]]
    live: list[Sink[LiveObservation]]
    resources: list[Resource]


def build_sinks(
    config: ExportersConfig | None,
    pipeline_index: int,
) -> SinkBundle:
    if config is None:
        return SinkBundle([], [], [])

    event_sinks: list[Sink[CompletedEvent]] = []
    workers: list[BufferedSink] = []
    for item in config_list(config.telegram):
        event_sinks.append(_buffered(item, TelegramSink(item), workers))
    for item in config_list(config.webhook):
        event_sinks.append(_buffered(item, WebhookSink(item), workers))
    for item in config_list(config.disk):
        event_sinks.append(_buffered(item, DiskSink(item), workers))

    streams: list[SSESink] = []
    for item in config_list(config.sse):
        stream = SSESink(
            replace(
                item,
                endpoint=item.endpoint or default_sse_endpoint(pipeline_index),
            )
        )
        streams.append(stream)
        event_sinks.append(_buffered(item, stream, workers))

    return SinkBundle(event_sinks, list(streams), [*workers, *streams])


def _buffered(
    config: ExporterConfig,
    sink: Sink[CompletedEvent],
    workers: list[BufferedSink],
) -> FilteredEventSink:
    worker = BufferedSink(sink)
    workers.append(worker)
    return FilteredEventSink(
        ExportPolicy(config.confidence or 0, config.export_rejected),
        worker,
    )
