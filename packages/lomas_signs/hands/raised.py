from __future__ import annotations

import numpy as np

from lomas_signs.hand import HAND_READERS
from lomas_signs.types import Box, Sign

RAISED = "raised_hand"
HALF = 2
FULL = 255
NEEDS_FACES = "raised_hand, which needs a face to look beside"


@HAND_READERS.register(RAISED)
class RaisedHand:
    """A hand up beside a face, found with no model at all.

    Written because the robot could not run the other one: mediapipe's
    aarch64 wheel is built for a processor with AES instructions, and a
    Raspberry Pi 4 has none - it does not fail, it aborts the whole process
    with an illegal instruction.

    It knows one thing, which is the only thing this robot ever needed a
    hand for: somebody would like to ask something. It cannot tell a thumb
    from a peace sign and does not try. What it costs is a colour threshold
    and a contour, a few milliseconds, and no download.

    It looks only above and beside a face the camera has already found, so
    the hand it finds belongs to somebody - and a face is never mistaken for
    a hand, because every face in the frame is cut out of the search first.
    """

    name = RAISED

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self._before: np.ndarray | None = None
        self._stirred: list[tuple[Box, float]] = []

    @property
    def available(self) -> bool:
        return True

    def describe(self) -> str:
        return NEEDS_FACES

    def close(self) -> None:
        return None

    def read(self, image: np.ndarray, at: float = 0.0,
             faces: tuple[Box, ...] = ()) -> list[Sign]:
        # No face, no hand: a hand on its own says nothing about who is
        # asking, and this reader has nowhere to look.
        if image is None or not faces:
            return []

        import cv2

        skin = self._skin(image, cv2)
        for face in faces:
            # Every face is taken out of the search, or a child behind
            # somebody's shoulder is their raised hand.
            self._cut_out(skin, face)

        self._remember_movement(image, at, cv2)

        found: list[Sign] = []
        for face in faces:
            hand = self._beside(skin, face, image.shape, cv2)
            if hand is not None and self._stirred_lately(hand, at):
                found.append(Sign(name=RAISED, box=hand, score=1.0, at=at))
        return found

    # --- a hand goes up; furniture does not --------------------------------

    def _remember_movement(self, image: np.ndarray, at: float, cv2) -> None:
        """Where the picture has changed since the last look.

        Skin colour on its own calls a wooden door, a beige wall and a
        cardboard box hands - and calls them hands in every frame for the
        whole afternoon, which is exactly what the robot reported. What
        separates a hand from a doorframe is that a hand arrived.
        """
        if not self.cfg.needs_motion:
            return

        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        before, self._before = self._before, grey
        self._stirred = [(box, seen) for box, seen in self._stirred
                         if at - seen <= self.cfg.motion_window_seconds]
        if before is None or before.shape != grey.shape:
            return

        moving = cv2.threshold(cv2.absdiff(grey, before), self.cfg.motion_threshold,
                               FULL, cv2.THRESH_BINARY)[1]
        contours, _ = cv2.findContours(moving, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            self._stirred.append((Box(x=x, y=y, w=w, h=h), at))

    def _stirred_lately(self, hand: Box, at: float) -> bool:
        """Whether anything moved where this hand is, recently enough.

        A window rather than this frame alone: a hand is raised and then
        held still, and a child holding their hand up patiently is the one
        this whole feature is for.
        """
        if not self.cfg.needs_motion:
            return True
        wanted = self.cfg.motion_fraction * hand.w * hand.h
        return any(at - seen <= self.cfg.motion_window_seconds
                   and _overlap(hand, box) >= wanted
                   for box, seen in self._stirred)

    # --- what skin looks like ---------------------------------------------

    def _skin(self, image: np.ndarray, cv2) -> np.ndarray:
        """Skin in chroma, not brightness.

        Cr and Cb hardly move across skin tones where brightness moves a
        great deal, which is what lets one setting hold for a whole
        classroom - and for the same child by the window and at the back.
        """
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        low = np.array([0, self.cfg.skin_cr[0], self.cfg.skin_cb[0]], np.uint8)
        high = np.array([FULL, self.cfg.skin_cr[1], self.cfg.skin_cb[1]], np.uint8)
        mask = cv2.inRange(ycrcb, low, high)
        return cv2.medianBlur(mask, self.cfg.blur_px | 1)

    def _cut_out(self, mask: np.ndarray, face: Box) -> None:
        mask[max(face.y, 0):face.y + face.h, max(face.x, 0):face.x + face.w] = 0

    # --- where a raised hand is -------------------------------------------

    def _beside(self, mask: np.ndarray, face: Box, shape, cv2) -> Box | None:
        face_area = float(face.w * face.h)
        for left, top, right, bottom in self._arch(face, shape):
            found = self._biggest(mask[top:bottom, left:right], face_area, cv2)
            if found is None:
                continue
            x, y, w, h = found
            return Box(x=left + x, y=top + y, w=w, h=h)
        return None

    def _arch(self, face: Box, shape) -> list[tuple[int, int, int, int]]:
        """Where a raised hand can be, and nowhere else.

        An arch over the head, not a box around it: above, and down each
        side, but never the strip directly under the chin. That strip is a
        neck and a chest, which are skin, are always there, and were read as
        a raised hand on nearly every frame of a whole class - two hundred
        and seventy-five of them.
        """
        height, width = shape[:2]
        top = max(int(face.y - face.h * self.cfg.above_face), 0)
        left = max(int(face.x - face.w * self.cfg.beside_face), 0)
        right = min(int(face.x + face.w * (1 + self.cfg.beside_face)), width)
        shoulder = min(int(face.y + face.h * (1 + self.cfg.below_face)), height)

        over = (left, top, right, face.y)
        beside_left = (left, face.y, face.x, shoulder)
        beside_right = (face.x + face.w, face.y, right, shoulder)
        return [where for where in (over, beside_left, beside_right)
                if where[2] > where[0] and where[3] > where[1]]

    def _biggest(self, patch: np.ndarray, face_area: float, cv2):
        contours, _ = cv2.findContours(patch, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        biggest = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(biggest))
        if not (self.cfg.min_area * face_area <= area <= self.cfg.max_area * face_area):
            return None
        return cv2.boundingRect(biggest)


def _overlap(one: Box, other: Box) -> float:
    """How much of the picture these two share."""
    across = min(one.x + one.w, other.x + other.w) - max(one.x, other.x)
    down = min(one.y + one.h, other.y + other.h) - max(one.y, other.y)
    return float(across * down) if across > 0 and down > 0 else 0.0
