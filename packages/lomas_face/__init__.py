from lomas_core.schema import (
    AttentionConfig,
    EnrolmentConfig,
    FaceConfig,
    PoseConfig,
    PrivacyConfig,
)
from lomas_face.attention import AttentionMonitor
from lomas_face.detector import DETECTORS, FaceDetector
from lomas_face.embedder import EMBEDDERS, FaceEmbedder, distance, normalise
from lomas_face.enrolment import EnrolmentResult, EnrolmentSession, FrameFeedback
from lomas_face.identity import IdentityMatcher
from lomas_face.pose import estimate as estimate_pose
from lomas_face.quality import angle_bucket, assess, crop_face, sharpness
from lomas_face.tracker import Tracker, iou
from lomas_face.types import Detection, Disengagement, Landmarks, Pose, Track

from lomas_face import detectors as _detectors  # noqa: F401
from lomas_face import embedders as _embedders  # noqa: F401

DETECTORS.discover("lomas_face.detectors")
EMBEDDERS.discover("lomas_face.embedders")

__all__ = [
    "DETECTORS",
    "EMBEDDERS",
    "AttentionConfig",
    "AttentionMonitor",
    "Detection",
    "Disengagement",
    "EnrolmentConfig",
    "EnrolmentResult",
    "EnrolmentSession",
    "FaceConfig",
    "FaceDetector",
    "FaceEmbedder",
    "FrameFeedback",
    "IdentityMatcher",
    "Landmarks",
    "Pose",
    "PoseConfig",
    "PrivacyConfig",
    "Track",
    "Tracker",
    "angle_bucket",
    "assess",
    "crop_face",
    "distance",
    "estimate_pose",
    "iou",
    "normalise",
    "sharpness",
]
