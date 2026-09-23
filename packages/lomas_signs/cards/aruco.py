from __future__ import annotations

import math

import numpy as np

from lomas_core.errors import LomasError
from lomas_signs.card import CARD_READERS
from lomas_signs.types import DEGREES, TURNS, Box, Card

ARUCO = "aruco"
QUARTER = DEGREES / TURNS
TOP_LEFT = 0
TOP_RIGHT = 1


@CARD_READERS.register(ARUCO)
class ArucoCards:
    """Printed square markers, read straight out of OpenCV.

    No new dependency and no model file: the dictionaries are built in. That
    is the whole reason cards come before hands - a card costs a couple of
    milliseconds a frame where a hand model costs tens, and the card says
    *who* is holding it, which a hand never does.

    Which way up the card is held is an answer: A, B, C or D by edge. A
    class of forty answers a question in one frame, with no microphone and
    nothing to transcribe.
    """

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self._detector = None

    @property
    def available(self) -> bool:
        return self._built() is not None

    def describe(self) -> str:
        return f"{ARUCO} {self.cfg.dictionary}" if self.available else "no card reader"

    def read(self, image: np.ndarray, at: float = 0.0) -> list[Card]:
        detector = self._built()
        if detector is None or image is None:
            return []

        corners, ids, _rejected = detector.detectMarkers(image)
        if ids is None:
            return []

        seen: list[Card] = []
        for marker, quad in zip(ids.ravel().tolist(), corners):
            points = quad.reshape(-1, 2)
            box = _around(points)
            if min(box.w, box.h) < self.cfg.min_size_px:
                # A speck across the room, or a pattern in somebody's shirt.
                continue
            seen.append(Card(marker_id=int(marker), box=box, turn=_turn(points), at=at))
        return seen

    def _built(self):
        if self._detector is not None:
            return self._detector
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - opencv is a hard dependency
            raise LomasError("opencv is not installed, so cards cannot be read") from exc

        family = getattr(cv2.aruco, self.cfg.dictionary, None)
        if family is None:
            raise LomasError(
                f"no marker dictionary '{self.cfg.dictionary}'. "
                "DICT_4X4_50 is the one the card sheet prints."
            )
        self._detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(family), cv2.aruco.DetectorParameters()
        )
        return self._detector


def _around(points: np.ndarray) -> Box:
    left, top = points.min(axis=0)
    right, bottom = points.max(axis=0)
    return Box(x=int(left), y=int(top), w=int(right - left), h=int(bottom - top))


def _turn(points: np.ndarray) -> int:
    """Which way up the card is being held.

    The detector always returns the marker's own top edge first, whatever
    the card is doing in the room - so the angle of that edge in the picture
    is the rotation, and a quarter turn is an answer.
    """
    top = points[TOP_RIGHT] - points[TOP_LEFT]
    angle = math.degrees(math.atan2(float(top[1]), float(top[0]))) % DEGREES
    return int(round(angle / QUARTER)) % TURNS
