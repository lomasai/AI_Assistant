from __future__ import annotations

import numpy as np

from lomas_signs.hand import HAND_READERS
from lomas_signs.types import Box, Sign

RAISED = "raised_hand"
HALF = 2
FULL = 255
HALF_LIT = FULL // HALF
ONE = 1.0
LEAST = 0.02   # a frame always teaches the room a little
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
        self._usual: np.ndarray | None = None
        self._usual_at = 0.0

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

        found: list[Sign] = []
        for face in faces:
            hand = self._beside(skin, face, image.shape, cv2)
            if hand is not None and self._is_new(hand):
                found.append(Sign(name=RAISED, box=hand, score=1.0, at=at))

        # Learned after the looking, so a hand that is up right now does not
        # teach the robot that its own corner of the room is furniture.
        self._learn_the_room(skin, at)
        return found

    # --- a hand arrives; a wall was always there ---------------------------

    def _learn_the_room(self, skin: np.ndarray, at: float) -> None:
        """Which parts of the picture are usually skin-coloured.

        Frame-to-frame movement was the first answer to "is this a hand or a
        doorframe" and a poor one: a Pi camera with auto-exposure changes
        every pixel a little, every frame, so everything moves and nothing
        is ruled out - the robot reported a raised hand on twenty reads out
        of twenty with nobody's hand up.

        This asks a better question. A wooden door is skin-coloured in every
        frame of the afternoon; a hand is skin-coloured in the few seconds
        it is up. So the robot keeps a running average of where skin is, and
        will not find a hand anywhere that is skin most of the time.
        """
        if not self.cfg.needs_motion:
            return

        lit = (skin > HALF_LIT).astype(np.float32)
        if self._usual is None or self._usual.shape != lit.shape:
            self._usual = lit
            self._usual_at = at
            return

        # How much of this frame to believe, by how long it has been. A
        # camera that drops frames learns at the same pace as one that does
        # not, which is what makes the setting a time and not a count.
        pace = (at - self._usual_at) / self.cfg.settles_in_seconds
        share = min(max(pace, LEAST), ONE)
        self._usual = self._usual * (ONE - share) + lit * share
        self._usual_at = at

    def _is_new(self, hand: Box) -> bool:
        """Whether this patch is skin more often than a hand ever would be."""
        if not self.cfg.needs_motion or self._usual is None:
            return True

        patch = self._usual[max(hand.y, 0):hand.y + hand.h, max(hand.x, 0):hand.x + hand.w]
        if not patch.size:
            return True
        return float(patch.mean()) < self.cfg.usually_skin_at

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
