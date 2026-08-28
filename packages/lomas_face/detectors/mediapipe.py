from __future__ import annotations

import numpy as np

from lomas_core.errors import LomasError
from lomas_core.schema import FaceConfig
from lomas_face.detector import DETECTORS
from lomas_face.types import Detection, Landmarks

KEYPOINTS = 6


@DETECTORS.register("mediapipe")
class MediaPipeDetector:
    """Alternative to YuNet. Heavier on a Pi 4 but ships its own model, so it
    is the easier one to get running the first time."""

    def __init__(self, cfg: FaceConfig) -> None:
        self.cfg = cfg
        self._detector = None

    def _ensure(self) -> None:
        if self._detector is not None:
            return
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise LomasError(
                "mediapipe is not installed. pip install mediapipe, or use "
                "face.detector: yunet."
            ) from exc
        self._detector = mp.solutions.face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=self.cfg.min_confidence
        )

    def detect(self, image: np.ndarray) -> list[Detection]:
        import cv2

        self._ensure()
        height, width = image.shape[:2]
        result = self._detector.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not result.detections:
            return []

        found: list[Detection] = []
        for item in result.detections:
            box = item.location_data.relative_bounding_box
            points = item.location_data.relative_keypoints
            landmarks = None
            if len(points) >= KEYPOINTS:
                landmarks = Landmarks(
                    right_eye=(points[0].x * width, points[0].y * height),
                    left_eye=(points[1].x * width, points[1].y * height),
                    nose=(points[2].x * width, points[2].y * height),
                    right_mouth=(points[3].x * width, points[3].y * height),
                    left_mouth=(points[3].x * width, points[3].y * height),
                )
            found.append(
                Detection(
                    x=max(0, int(box.xmin * width)),
                    y=max(0, int(box.ymin * height)),
                    w=max(1, int(box.width * width)),
                    h=max(1, int(box.height * height)),
                    confidence=float(item.score[0]),
                    landmarks=landmarks,
                )
            )
        return found
