from __future__ import annotations

import numpy as np

from lomas_face.types import Track

# The detector gives the two corners of the mouth and nothing else, so the
# mouth is taken as a box around them: this much of the distance between the
# corners, out to each side and above and below.
WIDTH_MARGIN = 0.35
HEIGHT_SHARE = 0.6
SMALLEST = 8  # a mouth smaller than this in pixels is noise, not a mouth
GREY = (0.114, 0.587, 0.299)  # BGR, as OpenCV hands it over
COLOUR_DIMS = 3  # a colour image, as against an already-grey one


def mouth_box(track: Track) -> tuple[int, int, int, int] | None:
    """Where the mouth is, from the two corners the detector found."""
    marks = track.box.landmarks
    if marks is None:
        return None

    left, right = marks.left_mouth, marks.right_mouth
    centre_x = (left[0] + right[0]) / 2
    centre_y = (left[1] + right[1]) / 2
    width = abs(left[0] - right[0]) * (1 + WIDTH_MARGIN * 2)
    height = max(width * HEIGHT_SHARE, SMALLEST)

    return (
        int(centre_x - width / 2), int(centre_y - height / 2),
        int(max(width, SMALLEST)), int(height),
    )


def patch(image: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = box
    height, width = image.shape[:2]
    cut = image[max(0, y): min(height, y + h), max(0, x): min(width, x + w)]
    if cut.size == 0:
        return cut
    grey = (cut @ np.array(GREY, dtype=np.float32) if cut.ndim == COLOUR_DIMS
            else cut.astype(np.float32))
    return grey / 255.0


class Mouths:
    """How much each face's mouth is moving.

    Not whether the mouth is open: the detector's five points are two eyes, a
    nose and the two corners of the mouth, so there is no top or bottom lip
    to measure against. What there is instead is movement - the pixels around
    the mouth change while someone is speaking and sit still when they are
    not - and for "which of these two is talking" that is the question
    anyway.

    Scores are smoothed over a few frames, because a single frame of a mouth
    mid-syllable looks exactly like a single frame of somebody turning their
    head.
    """

    def __init__(self, smoothing: float, size: int) -> None:
        self.smoothing = smoothing
        self.size = size
        self._last: dict[int, np.ndarray] = {}
        self._score: dict[int, float] = {}

    def update(self, image: np.ndarray, tracks: list[Track]) -> dict[int, float]:
        """One frame. Returns the movement score per track, 0 for a face
        whose mouth could not be found."""
        import cv2

        live = set()
        for track in tracks:
            live.add(track.track_id)
            box = mouth_box(track)
            crop = patch(image, box) if box else np.empty(0)
            if crop.size == 0:
                continue

            now = cv2.resize(crop, (self.size, self.size))
            before = self._last.get(track.track_id)
            self._last[track.track_id] = now
            if before is None:
                continue

            moved = float(np.abs(now - before).mean())
            was = self._score.get(track.track_id, 0.0)
            self._score[track.track_id] = was * self.smoothing + moved * (1 - self.smoothing)

        for gone in [at for at in self._last if at not in live]:
            self._last.pop(gone, None)
            self._score.pop(gone, None)
        return dict(self._score)

    def score(self, track_id: int) -> float:
        return self._score.get(track_id, 0.0)
