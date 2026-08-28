from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lomas_core.schema import EnrolmentConfig, PoseConfig
from lomas_face.pose import estimate
from lomas_face.types import Detection, Pose

CENTRE = "centre"
LEFT = "left"
RIGHT = "right"

FULL = 1.0
HALF = 2
COLOUR_CHANNELS = 3


@dataclass(frozen=True, slots=True)
class Quality:
    sharpness: float
    face_px: int
    angle: str
    pose: Pose | None

    @property
    def score(self) -> float:
        return self.sharpness


def sharpness(crop: np.ndarray, reference: float) -> float:
    """Laplacian variance, normalised against a configured reference.

    A blurred enrolment frame produces a vector that matches nobody later, so
    it is worth rejecting at capture time rather than discovering in class.
    """
    import cv2

    grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == COLOUR_CHANNELS else crop
    variance = float(cv2.Laplacian(grey, cv2.CV_64F).var())
    return min(FULL, variance / reference) if reference > 0 else FULL


def angle_bucket(pose: Pose | None, boundary: float) -> str:
    if pose is None:
        return CENTRE
    if pose.yaw <= -boundary:
        return RIGHT
    if pose.yaw >= boundary:
        return LEFT
    return CENTRE


def crop_face(image: np.ndarray, detection: Detection, margin: float) -> np.ndarray:
    """Cut the face out with a little context around it.

    Always call this against the full-resolution frame. Cropping from the
    downscaled copy used for detection throws away exactly the detail the
    embedder needs.
    """
    height, width = image.shape[:2]
    pad_x = int(detection.w * margin)
    pad_y = int(detection.h * margin)

    x1 = max(0, detection.x - pad_x)
    y1 = max(0, detection.y - pad_y)
    x2 = min(width, detection.x + detection.w + pad_x)
    y2 = min(height, detection.y + detection.h + pad_y)
    return image[y1:y2, x1:x2]


def assess(
    image: np.ndarray, detection: Detection, cfg: EnrolmentConfig, pose_cfg: PoseConfig
) -> Quality:
    crop = crop_face(image, detection, cfg.crop_margin)
    pose = estimate(detection.landmarks, pose_cfg) if detection.landmarks else None
    return Quality(
        sharpness=sharpness(crop, cfg.sharpness_reference),
        face_px=detection.w,
        angle=angle_bucket(pose, cfg.angle_yaw_degrees),
        pose=pose,
    )
