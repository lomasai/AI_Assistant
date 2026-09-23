from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from lomas_core.registry import Registry
from lomas_signs.types import Box, Sign


@runtime_checkable
class HandReader(Protocol):
    """One way of reading a hand held up in a frame.

    Returns what it saw and nothing about what it means. A reader that
    cannot run - no model file, no library installed, a wheel built for
    another processor - reports itself as unavailable rather than raising,
    because a robot with no hand reader is a robot that still teaches.

    `faces` is where the camera has already found faces, in this image's
    own pixels. A reader that recognises hands on their own ignores it; the
    one that looks beside a face needs it.
    """

    @property
    def available(self) -> bool: ...

    def read(self, image: np.ndarray, at: float = 0.0,
             faces: tuple[Box, ...] = ()) -> list[Sign]: ...

    def describe(self) -> str: ...

    def close(self) -> None: ...


HAND_READERS: Registry[HandReader] = Registry("hand reader")
