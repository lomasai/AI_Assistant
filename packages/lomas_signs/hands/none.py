from __future__ import annotations

import numpy as np

from lomas_signs.hand import HAND_READERS
from lomas_signs.types import Sign

NOTHING = "no hand reader"


@HAND_READERS.register("none")
class NoHands:
    """Reads no hands, on purpose.

    Hand reading is the one expensive thing in this feature - a model
    running on every frame of a Pi that is already busy. A school that finds
    it too dear sets this and keeps the cards, which cost almost nothing.
    """

    def __init__(self, cfg=None) -> None:
        self.cfg = cfg

    @property
    def available(self) -> bool:
        return False

    def read(self, image: np.ndarray, at: float = 0.0) -> list[Sign]:
        return []

    def describe(self) -> str:
        return NOTHING

    def close(self) -> None:
        return None
