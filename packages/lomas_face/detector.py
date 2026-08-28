from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from lomas_core.registry import Registry
from lomas_face.types import Detection


@runtime_checkable
class FaceDetector(Protocol):
    def detect(self, image: np.ndarray) -> list[Detection]: ...


DETECTORS: Registry[FaceDetector] = Registry("face detector")
