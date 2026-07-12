from threading import Lock

from aidetector.domain.events import DetectionEvent
from aidetector.media.rendering import (
    compress_jpg,
    frame_image,
    frame_jpg,
    get_crop,
    get_image,
    get_plot,
)
from aidetector.media.video import generate_mp4


class EventArtifacts:
    def __init__(self, event: DetectionEvent):
        self.event = event
        self._images: dict[tuple, bytes | None] = {}
        self._videos: dict[tuple, bytes | None] = {}
        self._lock = Lock()

    def image(
        self,
        *,
        plot: bool = False,
        crop: bool = False,
        padding: float = 0.1,
        data_max: int | None = None,
    ) -> bytes | None:
        key = (plot, crop, padding, data_max)
        with self._lock:
            if key not in self._images:
                observation = self.event.best
                if not plot and not crop and data_max is None:
                    encoded = frame_jpg(observation.frame)
                else:
                    if crop:
                        image = get_crop(observation, padding=padding, plot=plot)
                    elif plot:
                        image = get_plot(observation)
                    else:
                        image = frame_image(observation.frame)

                    if image is None:
                        encoded = None
                    elif data_max is not None:
                        encoded = compress_jpg(image, data_max)
                    else:
                        encoded = get_image(image)
                self._images[key] = encoded
            return self._images[key]

    def video(
        self,
        *,
        width: int | None = None,
        crf: int = 0,
        crop: bool = True,
        plot: bool = True,
        padding: float = 0.1,
        data_max: int | None = None,
    ) -> bytes | None:
        key = (width, crf, crop, plot, padding, data_max)
        with self._lock:
            if key not in self._videos:
                self._videos[key] = generate_mp4(
                    self.event.observations,
                    width=width,
                    crf=crf,
                    crop=crop,
                    plot=plot,
                    padding=padding,
                    data_max=data_max,
                )
            return self._videos[key]
