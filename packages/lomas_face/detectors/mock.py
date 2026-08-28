from __future__ import annotations

import numpy as np

from lomas_core.schema import FaceConfig
from lomas_face.detector import DETECTORS
from lomas_face.types import Detection


@DETECTORS.register("mock")
class MockDetector:
    """Replays a scripted list of detections, one entry per detect() call.

    Tracking and attention are state machines, and state machines are worth
    testing without a model in the way. Load a script and every run is
    identical.
    """

    def __init__(self, cfg: FaceConfig | None = None) -> None:
        self.cfg = cfg
        self._script: list[list[Detection]] = []
        self._index = 0
        self.calls = 0

    def script(self, frames: list[list[Detection]]) -> None:
        self._script = frames
        self._index = 0

    def detect(self, image: np.ndarray | None = None) -> list[Detection]:
        self.calls += 1
        if self._index >= len(self._script):
            return []
        frame = self._script[self._index]
        self._index += 1
        return frame
