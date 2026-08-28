from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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
