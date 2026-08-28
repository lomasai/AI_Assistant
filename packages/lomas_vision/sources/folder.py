from __future__ import annotations

from pathlib import Path

import numpy as np

from lomas_core.errors import LomasError
from lomas_core.schema import SourceConfig
from lomas_vision.source import CAMERA_SOURCES, BaseSource

SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp")


@CAMERA_SOURCES.register("folder")
class FolderSource(BaseSource):
    """Still images in filename order. Fully deterministic, which is what
    makes recognition regressions reproducible."""

    def __init__(self, spec: SourceConfig) -> None:
        super().__init__(spec)
        self._paths: list[Path] = []
        self._index = 0

    def _start(self) -> None:
        if not self.spec.path:
            raise LomasError(f"source '{self.source_id}': folder kind needs a path")
        root = Path(self.spec.path)
        if not root.is_dir():
            raise LomasError(f"source '{self.source_id}': {root} is not a directory")
        self._paths = sorted(p for p in root.iterdir() if p.suffix.lower() in SUFFIXES)
        if not self._paths:
            raise LomasError(f"source '{self.source_id}': no images in {root}")

    def _grab(self) -> np.ndarray | None:
        if self._index >= len(self._paths):
            if not self.spec.loop:
                return None
            self._index = 0

        import cv2

        image = cv2.imread(str(self._paths[self._index]))
        self._index += 1
        return image
