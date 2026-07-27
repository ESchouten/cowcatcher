from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

from aidetector.domain.detections import Observation
from aidetector.domain.events import DetectionEvent
from aidetector.domain.frames import Frame, FrameBatch
from numpy import ndarray

T = TypeVar("T", contravariant=True)


class Sink(Protocol[T]):
    def send(self, message: T) -> None: ...


class Resource(Protocol):
    def start(self) -> None: ...

    def close(self) -> None: ...


class FrameSource(Protocol):
    sources: list[str]

    @property
    def realtime(self) -> bool: ...

    def iter_batches(self) -> Iterator[FrameBatch]: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class ModelBatchResult:
    source: str
    result: Any
    frames: list[Frame]


class ModelRunner(Protocol):
    def detect(self, frames: list[ndarray]) -> list[Any]: ...

    def track_sources(self, batch: FrameBatch) -> list[ModelBatchResult]: ...

    def observations_from_result(
        self,
        result: Any,
        frames: list[Frame],
    ) -> list[Observation] | None: ...


class ObservationEnricher(Protocol):
    def enrich(
        self,
        source: str,
        observation: Observation,
        /,
    ) -> Observation: ...


class ArtifactProvider(Protocol):
    def image(
        self,
        *,
        plot: bool = False,
        crop: bool = False,
        padding: float = 0.1,
        data_max: int | None = None,
    ) -> bytes | None: ...

    def video(
        self,
        *,
        width: int | None = None,
        crf: int = 0,
        crop: bool = True,
        plot: bool = True,
        padding: float = 0.1,
        data_max: int | None = None,
    ) -> bytes | None: ...


class EventValidator(Protocol):
    def validate(
        self,
        event: DetectionEvent,
        artifacts: ArtifactProvider,
    ) -> bool | None: ...


ArtifactFactory = Callable[[DetectionEvent], ArtifactProvider]
