from __future__ import annotations

import time
from typing import Protocol, runtime_checkable

import numpy as np

from lomas_core.registry import Registry
from lomas_core.schema import SourceConfig
from lomas_vision.frame import Frame

QUARTER_TURN = 90
UNPACED = 0


@runtime_checkable
class CameraSource(Protocol):
    source_id: str
    zone: str

    def open(self) -> None: ...

    def read(self) -> Frame | None: ...

    def set_zoom(self, factor: float) -> None: ...

    def close(self) -> None: ...


CAMERA_SOURCES: Registry[CameraSource] = Registry("camera source")


class BaseSource:
    """Shared bookkeeping. Subclasses implement `_grab` and optionally
    `_start` / `_stop`.

    Synthetic sources return instantly, so `read` paces itself to the
    configured fps. A real camera already blocks for roughly that long, which
    makes the pacing a no-op there. `fps: 0` disables it entirely, which is
    what the tests use so a hundred frames take milliseconds.
    """

    def __init__(self, spec: SourceConfig) -> None:
        self.spec = spec
        self.source_id = spec.id
        self.zone = spec.zone
        self.zoom = spec.zoom
        self._seq = 0
        self._opened = False
        self._interval = 1.0 / spec.fps if spec.fps > UNPACED else 0.0
        self._next_due = 0.0

    def open(self) -> None:
        if not self._opened:
            self._start()
            self._opened = True

    def read(self) -> Frame | None:
        if not self._opened:
            self.open()

        self._pace()
        image = self._grab()
        if image is None:
            return None

        self._seq += 1
        return Frame(
            source_id=self.source_id,
            zone=self.zone,
            seq=self._seq,
            ts=time.monotonic(),
            image=rotate(image, self.spec.rotation),
        )

    def set_zoom(self, factor: float) -> None:
        self.zoom = factor

    def close(self) -> None:
        if self._opened:
            self._stop()
            self._opened = False

    def _pace(self) -> None:
        if not self._interval:
            return
        now = time.monotonic()
        remaining = self._next_due - now
        if remaining > 0:
            time.sleep(remaining)
            now = self._next_due
        self._next_due = now + self._interval

    def _start(self) -> None: ...

    def _stop(self) -> None: ...

    def _grab(self) -> np.ndarray | None:
        raise NotImplementedError


def rotate(image: np.ndarray, degrees: int) -> np.ndarray:
    if not degrees:
        return image
    return np.rot90(image, k=degrees // QUARTER_TURN).copy()
