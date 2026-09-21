from __future__ import annotations

import numpy as np

from lomas_face.quality import crop_face
from lomas_face.types import Detection

# What SFace's own alignment expects, in this order: the box, then the five
# points the detector already gives, then the score.
ROW_LENGTH = 15


def as_row(detection: Detection) -> np.ndarray | None:
    """One detection in the shape OpenCV's aligner wants, or None when the
    detector gave no landmarks to align on."""
    marks = detection.landmarks
    if marks is None:
        return None

    return np.array([[
        detection.x, detection.y, detection.w, detection.h,
        *marks.right_eye, *marks.left_eye, *marks.nose,
        *marks.right_mouth, *marks.left_mouth,
        detection.confidence,
    ]], dtype=np.float32)


def face_for(embedder, image: np.ndarray, detection: Detection, margin: float,
             align: bool) -> np.ndarray:
    """The picture of a face to embed, straightened when that is possible.

    One function for recognition and for enrolment, because the two have to
    agree: a vector stored from a plain crop and compared against a
    straightened one is a child the robot does not recognise.

    Straightening rotates and scales the face so the eyes land in the same
    place every time, which is what the model was trained on. A face turned
    to the side, or a detector with no landmarks, falls back to the crop.
    """
    if align and hasattr(embedder, "align"):
        row = as_row(detection)
        if row is not None:
            aligned = embedder.align(image, row)
            if aligned is not None and aligned.size:
                return aligned
    return crop_face(image, detection, margin)
