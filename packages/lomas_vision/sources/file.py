from __future__ import annotations

import numpy as np

from lomas_core.errors import LomasError
from lomas_vision.source import CAMERA_SOURCES
from lomas_vision.sources.capture import OpenCvSource


@CAMERA_SOURCES.register("file")
class FileSource(OpenCvSource):
    """Replays a recorded video.

    This is the backend that lets the whole vision pipeline be developed and
    regression-tested on a laptop: record one classroom clip, and every run
    against it is reproducible.
    """

    def _target(self) -> str:
        if not self.spec.path:
            raise LomasError(f"source '{self.source_id}': file kind needs a path")
        return self.spec.path

    def _configure(self) -> None:
        return  # a file has its own dimensions; do not fight them

    def _grab(self) -> np.ndarray | None:
        image = super()._grab()
        if image is not None or not self.spec.loop:
            return image

        import cv2

        self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return super()._grab()
