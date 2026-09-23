from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from lomas_core.registry import Registry
from lomas_signs.types import Sign


@runtime_checkable
class HandReader(Protocol):
    """One way of reading a hand held up in a frame.

    Returns what it saw and nothing about what it means. A reader that
    cannot run - no model file, no library installed - reports itself as
    unavailable rather than raising, because a robot with no hand reader is
    a robot that still teaches.
    """

    @property
    def available(self) -> bool: ...

    def read(self, image: np.ndarray, at: float = 0.0) -> list[Sign]: ...

    def describe(self) -> str: ...

    def close(self) -> None: ...


HAND_READERS: Registry[HandReader] = Registry("hand reader")
