"""Face detection and tracking module for edge device."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


class TrackingError(Exception):
    """Raised when detection/tracking cannot be performed."""


TrackingBackend = Literal["opencv", "mediapipe", "auto"]


@dataclass(slots=True)
class TrackingConfig:
    """Configuration for face detection and smoothing."""

    backend: TrackingBackend = "auto"
    min_confidence: float = 0.5
    smoothing_alpha: float = 0.35


@dataclass(slots=True)
class FaceBox:
    """Bounding box for detected face in pixel coordinates."""

    x: int
    y: int
    w: int
    h: int
    confidence: float = 1.0


@dataclass(slots=True)
class TrackResult:
    """Primary tracking output."""

    found: bool
    face_count: int
    bbox: FaceBox | None
    center_x: float | None
    center_y: float | None
    normalized_x: float | None
    normalized_y: float | None


class FaceTracker:
    """Detect and track face position with OpenCV/MediaPipe backends."""

    def __init__(self, config: TrackingConfig | None = None) -> None:
        self.config = config or TrackingConfig()
        self._last_center: tuple[float, float] | None = None
        self._backend: Literal["opencv", "mediapipe"] | None = None
        self._opencv_cascade: Any = None
        self._mediapipe_face_detection: Any = None

    def detect_faces(self, frame: Any) -> list[FaceBox]:
        """Detect faces in a single frame and return bounding boxes."""
        backend = self._resolve_backend()
        if backend == "mediapipe":
            return self._detect_with_mediapipe(frame)
        return self._detect_with_opencv(frame)

    def track(self, frame: Any) -> TrackResult:
        """Track primary face and return coordinates."""
        if frame is None:
            raise TrackingError("Frame is required for tracking.")

        faces = self.detect_faces(frame)
        if not faces:
            self._last_center = None
            return TrackResult(
                found=False,
                face_count=0,
                bbox=None,
                center_x=None,
                center_y=None,
                normalized_x=None,
                normalized_y=None,
            )

        primary = self._select_primary_face(faces)
        cx = primary.x + (primary.w / 2.0)
        cy = primary.y + (primary.h / 2.0)
        smooth_cx, smooth_cy = self._smooth_center(cx, cy)

        height, width = self._frame_shape(frame)
        if width <= 0 or height <= 0:
            raise TrackingError("Invalid frame dimensions.")

        nx = smooth_cx / float(width)
        ny = smooth_cy / float(height)

        return TrackResult(
            found=True,
            face_count=len(faces),
            bbox=primary,
            center_x=round(smooth_cx, 2),
            center_y=round(smooth_cy, 2),
            normalized_x=round(nx, 6),
            normalized_y=round(ny, 6),
        )

    def to_coordinates(self, result: TrackResult) -> dict[str, float | int | bool | None]:
        """Return compact coordinate output for controllers/API use."""
        return {
            "found": result.found,
            "face_count": result.face_count,
            "x": result.center_x,
            "y": result.center_y,
            "nx": result.normalized_x,
            "ny": result.normalized_y,
        }

    def _resolve_backend(self) -> Literal["opencv", "mediapipe"]:
        if self._backend is not None:
            return self._backend

        preferred = self.config.backend
        if preferred in {"mediapipe", "auto"}:
            try:
                import mediapipe as mp  # type: ignore

                self._mediapipe_face_detection = mp.solutions.face_detection.FaceDetection(
                    model_selection=0,
                    min_detection_confidence=self.config.min_confidence,
                )
                self._backend = "mediapipe"
                return self._backend
            except Exception:  # noqa: BLE001
                if preferred == "mediapipe":
                    raise TrackingError("MediaPipe backend requested but unavailable.")

        try:
            import cv2  # type: ignore

            self._opencv_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            if self._opencv_cascade.empty():
                raise TrackingError("Failed to load OpenCV Haar cascade.")
        except Exception as exc:  # noqa: BLE001
            raise TrackingError("OpenCV backend unavailable for face detection.") from exc

        self._backend = "opencv"
        return self._backend

    def _detect_with_opencv(self, frame: Any) -> list[FaceBox]:
        try:
            import cv2  # type: ignore
        except ImportError as exc:
            raise TrackingError("OpenCV is required for OpenCV tracking backend.") from exc

        if self._opencv_cascade is None:
            self._resolve_backend()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = self._opencv_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40),
        )
        return [FaceBox(x=int(x), y=int(y), w=int(w), h=int(h), confidence=1.0) for (x, y, w, h) in detections]

    def _detect_with_mediapipe(self, frame: Any) -> list[FaceBox]:
        try:
            import cv2  # type: ignore
        except ImportError as exc:
            raise TrackingError("OpenCV is required for MediaPipe frame conversion.") from exc

        if self._mediapipe_face_detection is None:
            self._resolve_backend()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._mediapipe_face_detection.process(rgb)
        if not result.detections:
            return []

        height, width = self._frame_shape(frame)
        faces: list[FaceBox] = []
        for detection in result.detections:
            box = detection.location_data.relative_bounding_box
            x = max(0, int(box.xmin * width))
            y = max(0, int(box.ymin * height))
            w = max(1, int(box.width * width))
            h = max(1, int(box.height * height))
            faces.append(FaceBox(x=x, y=y, w=w, h=h, confidence=float(detection.score[0])))
        return faces

    @staticmethod
    def _select_primary_face(faces: list[FaceBox]) -> FaceBox:
        # Prefer largest detected face (closest subject).
        return max(faces, key=lambda face: face.w * face.h)

    def _smooth_center(self, x: float, y: float) -> tuple[float, float]:
        if self._last_center is None:
            self._last_center = (x, y)
            return x, y

        alpha = min(max(self.config.smoothing_alpha, 0.0), 1.0)
        prev_x, prev_y = self._last_center
        sx = (alpha * x) + ((1.0 - alpha) * prev_x)
        sy = (alpha * y) + ((1.0 - alpha) * prev_y)
        self._last_center = (sx, sy)
        return sx, sy

    @staticmethod
    def _frame_shape(frame: Any) -> tuple[int, int]:
        shape = getattr(frame, "shape", None)
        if not shape or len(shape) < 2:
            raise TrackingError("Frame must provide a valid shape attribute.")
        height = int(shape[0])
        width = int(shape[1])
        return height, width


face_tracker = FaceTracker()
