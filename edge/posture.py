"""Posture detection module for edge device.

Supports:
- Sitting detection
- Standing detection

Uses simple landmark-based rules and optional MediaPipe frame inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


PostureLabel = Literal["sitting", "standing", "unknown"]


class PostureError(Exception):
    """Raised when posture detection fails."""


@dataclass(slots=True)
class PosePoint:
    """Normalized 2D pose point."""

    x: float
    y: float
    visibility: float = 1.0


@dataclass(slots=True)
class PostureResult:
    """Output for posture detection."""

    posture: PostureLabel
    confidence: float
    metadata: dict[str, float]


@dataclass(slots=True)
class PostureConfig:
    """Config values for rule-based posture detection."""

    visibility_threshold: float = 0.45
    standing_hip_knee_dy_min: float = 0.12
    sitting_hip_knee_dy_max: float = 0.09
    torso_upright_dx_max: float = 0.17
    use_mediapipe: bool = True
    mediapipe_min_confidence: float = 0.5


class PostureDetector:
    """Rule-based posture detector with optional MediaPipe extraction."""

    def __init__(self, config: PostureConfig | None = None) -> None:
        self.config = config or PostureConfig()
        self._mp_pose = None
        self._pose_estimator = None

    def detect_from_landmarks(
        self,
        left_shoulder: PosePoint,
        right_shoulder: PosePoint,
        left_hip: PosePoint,
        right_hip: PosePoint,
        left_knee: PosePoint,
        right_knee: PosePoint,
    ) -> PostureResult:
        """Detect posture from required upper/lower body landmarks."""
        min_visibility = min(
            left_shoulder.visibility,
            right_shoulder.visibility,
            left_hip.visibility,
            right_hip.visibility,
            left_knee.visibility,
            right_knee.visibility,
        )
        if min_visibility < self.config.visibility_threshold:
            return PostureResult(
                posture="unknown",
                confidence=max(0.0, min_visibility),
                metadata={"min_visibility": min_visibility},
            )

        hip_y = (left_hip.y + right_hip.y) / 2.0
        knee_y = (left_knee.y + right_knee.y) / 2.0
        shoulder_x = (left_shoulder.x + right_shoulder.x) / 2.0
        hip_x = (left_hip.x + right_hip.x) / 2.0

        hip_knee_dy = knee_y - hip_y
        torso_dx = abs(shoulder_x - hip_x)

        posture: PostureLabel = "unknown"
        confidence = 0.55

        # Basic posture rules:
        # - Standing: clear vertical hip->knee separation + torso mostly upright.
        if hip_knee_dy >= self.config.standing_hip_knee_dy_min and torso_dx <= self.config.torso_upright_dx_max:
            posture = "standing"
            confidence = min(0.99, 0.65 + ((hip_knee_dy - self.config.standing_hip_knee_dy_min) * 1.8))

        # - Sitting: compressed hip->knee vertical gap, often due to bent knees.
        elif hip_knee_dy <= self.config.sitting_hip_knee_dy_max:
            posture = "sitting"
            confidence = min(0.99, 0.65 + ((self.config.sitting_hip_knee_dy_max - hip_knee_dy) * 2.2))

        return PostureResult(
            posture=posture,
            confidence=round(max(0.0, confidence), 4),
            metadata={
                "hip_knee_dy": round(hip_knee_dy, 6),
                "torso_dx": round(torso_dx, 6),
                "min_visibility": round(min_visibility, 6),
            },
        )

    def detect_from_frame(self, frame_bgr: Any) -> PostureResult:
        """Detect posture from a frame using MediaPipe landmarks."""
        if frame_bgr is None:
            raise PostureError("Frame is required.")
        if not self.config.use_mediapipe:
            raise PostureError("MediaPipe detection is disabled in config.")

        pose_landmarks = self._extract_pose_landmarks(frame_bgr)
        if pose_landmarks is None:
            return PostureResult(posture="unknown", confidence=0.0, metadata={"reason": 0.0})

        # MediaPipe landmark indices used:
        # 11 left_shoulder, 12 right_shoulder, 23 left_hip, 24 right_hip, 25 left_knee, 26 right_knee
        try:
            left_shoulder = self._to_point(pose_landmarks[11])
            right_shoulder = self._to_point(pose_landmarks[12])
            left_hip = self._to_point(pose_landmarks[23])
            right_hip = self._to_point(pose_landmarks[24])
            left_knee = self._to_point(pose_landmarks[25])
            right_knee = self._to_point(pose_landmarks[26])
        except (IndexError, TypeError) as exc:
            raise PostureError(f"Incomplete pose landmarks: {exc}") from exc

        return self.detect_from_landmarks(
            left_shoulder=left_shoulder,
            right_shoulder=right_shoulder,
            left_hip=left_hip,
            right_hip=right_hip,
            left_knee=left_knee,
            right_knee=right_knee,
        )

    def _extract_pose_landmarks(self, frame_bgr: Any) -> Any | None:
        """Run MediaPipe Pose and return raw landmark list."""
        try:
            import cv2  # type: ignore
            import mediapipe as mp  # type: ignore
        except ImportError as exc:
            raise PostureError("`opencv-python` and `mediapipe` are required for frame-based posture detection.") from exc

        if self._pose_estimator is None:
            self._mp_pose = mp.solutions.pose
            self._pose_estimator = self._mp_pose.Pose(
                static_image_mode=False,
                min_detection_confidence=self.config.mediapipe_min_confidence,
                min_tracking_confidence=self.config.mediapipe_min_confidence,
            )

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._pose_estimator.process(rgb)
        if not result.pose_landmarks:
            return None
        return result.pose_landmarks.landmark

    @staticmethod
    def _to_point(landmark: Any) -> PosePoint:
        return PosePoint(
            x=float(landmark.x),
            y=float(landmark.y),
            visibility=float(getattr(landmark, "visibility", 1.0)),
        )


posture_detector = PostureDetector()
