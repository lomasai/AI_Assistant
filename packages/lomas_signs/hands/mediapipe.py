from __future__ import annotations

from pathlib import Path

import numpy as np

from lomas_core import logging as log
from lomas_signs.hand import HAND_READERS
from lomas_signs.types import Box, Sign

MEDIAPIPE = "mediapipe"
NONE_FOUND = "none"
MILLISECONDS = 1000.0
FIRST = 0


@HAND_READERS.register(MEDIAPIPE)
class MediapipeHands:
    """Hand shapes, from the recognizer Google ships pre-trained.

    Thumb up, thumb down, victory, pointing up, open palm, closed fist and
    one more come out of the box, so nothing here is trained or collected -
    which matters, because a gesture model trained on adults in an office
    does not survive a classroom.

    This is the expensive half of the feature: a model on every frame it is
    given, on a Pi that is already running a face detector. So it is given
    few frames - an interrupt is *held*, and a sign held for a second is
    caught at three frames a second as surely as at thirty.

    Needs `pip install mediapipe` and the .task file; without either it
    reports itself unavailable and the robot carries on with cards.
    """

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.log = log.get("signs")
        self._reader = None
        self._broken = False

    @property
    def available(self) -> bool:
        return self._built() is not None

    def describe(self) -> str:
        return f"{MEDIAPIPE} {Path(self.cfg.model).name}" if self.available else "no hand reader"

    def read(self, image: np.ndarray, at: float = 0.0) -> list[Sign]:
        reader = self._built()
        if reader is None or image is None:
            return []
        try:
            import mediapipe as mp

            frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=_as_rgb(image))
            found = reader.recognize(frame)
        except Exception as exc:  # the wheel raises its own types
            self.log.debug("hand reader skipped a frame: %s", exc)
            return []

        signs: list[Sign] = []
        height, width = image.shape[:2]
        for gestures, marks in zip(found.gestures, found.hand_landmarks):
            if not gestures:
                continue
            best = gestures[FIRST]
            if best.category_name == NONE_FOUND or best.score < self.cfg.min_score:
                continue
            signs.append(Sign(name=best.category_name, box=_around(marks, width, height),
                              score=float(best.score), at=at))
        return signs

    def close(self) -> None:
        if self._reader is not None:
            self._reader.close()
            self._reader = None

    def _built(self):
        if self._reader is not None or self._broken:
            return self._reader
        try:
            from mediapipe.tasks import python as tasks
            from mediapipe.tasks.python import vision
        except ImportError:
            # Said once, at the level a robot owner can act on. Cards still
            # work, and they are the cheap half anyway.
            self._refuse("mediapipe is not installed (pip install mediapipe)")
            return None

        model = Path(self.cfg.model)
        if not model.exists():
            self._refuse(f"no gesture model at {model} (python tools/fetch_models.py)")
            return None

        options = vision.GestureRecognizerOptions(
            base_options=tasks.BaseOptions(model_asset_path=str(model)),
            num_hands=self.cfg.max_hands,
            min_hand_detection_confidence=self.cfg.min_score,
        )
        self._reader = vision.GestureRecognizer.create_from_options(options)
        return self._reader

    def _refuse(self, why: str) -> None:
        self._broken = True
        self.log.info("hands are off: %s", why)


def _as_rgb(image: np.ndarray) -> np.ndarray:
    """Cameras here hand out BGR; mediapipe wants RGB and will happily read
    a blue child otherwise."""
    return np.ascontiguousarray(image[:, :, ::-1])


def _around(marks, width: int, height: int) -> Box:
    xs = [point.x * width for point in marks]
    ys = [point.y * height for point in marks]
    return Box(x=int(min(xs)), y=int(min(ys)), w=int(max(xs) - min(xs)), h=int(max(ys) - min(ys)))
