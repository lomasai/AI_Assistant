from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NO_SCALE = 1.0


@dataclass(slots=True)
class Frame:
    """One image with enough provenance to survive several cameras.

    `zone` travels with the frame so a consumer can tell the front-of-class
    camera from a back-wall one without looking up the source config.
    """

    source_id: str
    zone: str
    seq: int
    ts: float
    image: np.ndarray

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    @property
    def size(self) -> tuple[int, int]:
        return self.width, self.height


def downscale(image: np.ndarray, width: int) -> tuple[np.ndarray, float]:
    """A smaller copy, and the factor that maps a box back to the original.

    Everything that looks for something in a frame looks at a small copy and
    then works on the big one: a detector at 640 costs a quarter of what it
    costs at 1280 and finds the same faces. The factor is what lets the
    finding be drawn, cropped or measured in full-resolution pixels.
    """
    height, full_width = image.shape[:2]
    if full_width <= width:
        return image, NO_SCALE

    import cv2  # only needed when a frame is actually bigger than the target

    factor = full_width / width
    small = cv2.resize(image, (width, int(height / factor)), interpolation=cv2.INTER_AREA)
    return small, factor
