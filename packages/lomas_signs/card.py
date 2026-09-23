from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from lomas_core.registry import Registry
from lomas_signs.types import Card


@runtime_checkable
class CardReader(Protocol):
    """One way of reading a printed card held up in a frame."""

    @property
    def available(self) -> bool: ...

    def read(self, image: np.ndarray, at: float = 0.0) -> list[Card]: ...

    def describe(self) -> str: ...


CARD_READERS: Registry[CardReader] = Registry("card reader")
