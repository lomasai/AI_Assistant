from __future__ import annotations

import numpy as np

from lomas_signs.card import CARD_READERS
from lomas_signs.types import Card

NOTHING = "no card reader"


@CARD_READERS.register("none")
class NoCards:
    def __init__(self, cfg=None) -> None:
        self.cfg = cfg

    @property
    def available(self) -> bool:
        return False

    def read(self, image: np.ndarray, at: float = 0.0) -> list[Card]:
        return []

    def describe(self) -> str:
        return NOTHING
