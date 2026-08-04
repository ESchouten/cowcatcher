from dataclasses import dataclass, field
from datetime import datetime

from numpy import ndarray


@dataclass(frozen=True, slots=True)
class Frame:
    captured_at: datetime
    image: ndarray | None = field(repr=False, compare=False, default=None)
    jpeg: bytes | None = field(repr=False, compare=False, default=None)

    def __post_init__(self) -> None:
        if (self.image is None) == (self.jpeg is None):
            raise ValueError("Frame requires either image pixels or JPEG data")

    def require_image(self) -> ndarray:
        if self.image is None:
            raise RuntimeError("Frame pixels are not available")
        return self.image


FrameBatch = dict[str, list[Frame]]
