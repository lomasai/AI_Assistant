from __future__ import annotations

import math

from lomas_core.schema import PoseConfig
from lomas_face.types import Landmarks, Pose

FLAT = 0.0
FULL = 1.0


def clamp(value: float, lowest: float = -FULL, highest: float = FULL) -> float:
    return max(lowest, min(highest, value))


def estimate(landmarks: Landmarks, cfg: PoseConfig) -> Pose | None:
    """Head orientation from the five landmarks, no second model.

    This is a geometric approximation, not a calibrated 3D pose. It is good
    enough for the only question being asked - is this child roughly facing
    the robot - and it costs nothing, which a real pose model would not.
    """
    eye_x, eye_y = landmarks.eye_centre
    mouth_x, mouth_y = landmarks.mouth_centre
    nose_x, nose_y = landmarks.nose

    eye_span = math.dist(landmarks.right_eye, landmarks.left_eye)
    face_height = abs(mouth_y - eye_y)
    if eye_span <= FLAT or face_height <= FLAT:
        return None

    # Nose drifts toward the ear you are turning away from.
    yaw_ratio = clamp((nose_x - eye_x) / (eye_span / 2))
    yaw = yaw_ratio * cfg.yaw_scale_degrees

    # Nose rides up toward the eyes when the chin lifts.
    pitch_ratio = clamp(((nose_y - eye_y) / face_height) - cfg.pitch_neutral)
    pitch = pitch_ratio * cfg.pitch_scale_degrees

    roll = math.degrees(
        math.atan2(
            landmarks.left_eye[1] - landmarks.right_eye[1],
            landmarks.left_eye[0] - landmarks.right_eye[0],
        )
    )

    return Pose(yaw=yaw, pitch=pitch, roll=roll)
