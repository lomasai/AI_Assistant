from __future__ import annotations

import numpy as np

from lomas_core.errors import LomasError
from lomas_core.schema import SourceConfig
from lomas_vision.source import BaseSource


class OpenCvSource(BaseSource):
    """Shared behaviour for anything cv2.VideoCapture can open: a webcam, a
    video file, or an RTSP stream. Subclasses only supply the target."""

    def __init__(self, spec: SourceConfig) -> None:
        super().__init__(spec)
        self._capture = None

    def _target(self) -> int | str:
        raise NotImplementedError

    def _configure(self) -> None:
        import cv2

        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.spec.width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.spec.height)
        if self.spec.fps:
            self._capture.set(cv2.CAP_PROP_FPS, self.spec.fps)

    def _start(self) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise LomasError("opencv is required for this camera source") from exc

        target = self._target()
        self._capture = cv2.VideoCapture(target)
        if not self._capture.isOpened():
            raise LomasError(f"source '{self.source_id}': cannot open {target}")
        self._configure()

    def _grab(self) -> np.ndarray | None:
        ok, image = self._capture.read()
        if not ok:
            return None
        return image

    def _stop(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
