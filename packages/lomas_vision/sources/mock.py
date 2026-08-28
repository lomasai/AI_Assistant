from __future__ import annotations

import numpy as np

from lomas_core.schema import SourceConfig
from lomas_vision.source import CAMERA_SOURCES, BaseSource

CHANNELS = 3
BYTE_RANGE = 256
BLOCK_DIVISOR = 8


@CAMERA_SOURCES.register("mock")
class MockSource(BaseSource):
    """Synthetic frames with no dependencies beyond numpy.

    Each frame carries its sequence number in the pixel values and moves a
    block across the image, so a test can prove frames are distinct and
    ordered without decoding anything.
    """

    def __init__(self, spec: SourceConfig) -> None:
        super().__init__(spec)
        self._canvas = np.zeros((spec.height, spec.width, CHANNELS), dtype=np.uint8)

    def _grab(self) -> np.ndarray:
        frame = self._canvas.copy()
        frame[:, :] = self._seq % BYTE_RANGE

        block = max(1, self.spec.width // BLOCK_DIVISOR)
        left = (self._seq * block) % max(1, self.spec.width - block)
        frame[0:block, left : left + block] = BYTE_RANGE - 1
        return frame
