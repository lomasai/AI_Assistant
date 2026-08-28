from __future__ import annotations

from pathlib import Path

import numpy as np

from lomas_core.errors import LomasError
from lomas_core.schema import FaceConfig
from lomas_face.detector import DETECTORS
from lomas_face.types import Detection, Landmarks

BOX_FIELDS = 4
LANDMARK_FIELDS = 10
SCORE_INDEX = 14


@DETECTORS.register("yunet")
class YuNetDetector:
    """OpenCV's YuNet. Fast enough on a Pi 4 at around 5-10ms, and unlike a
    Haar cascade it survives a child turning their head."""

    def __init__(self, cfg: FaceConfig) -> None:
        self.cfg = cfg
        self._net = None
        self._size: tuple[int, int] | None = None

    def _ensure(self, width: int, height: int) -> None:
        import cv2

        if self._net is None:
            model = Path(self.cfg.model_path)
            if not model.exists():
                raise LomasError(
                    f"YuNet model not found at {model}. Download "
                    "face_detection_yunet_2023mar.onnx from the OpenCV Zoo and "
                    "point face.model_path at it, or use face.detector: mock."
                )
            self._net = cv2.FaceDetectorYN.create(
                str(model), "", (width, height),
                self.cfg.min_confidence, self.cfg.nms_threshold, self.cfg.top_k,
            )
            self._size = (width, height)
        elif self._size != (width, height):
            self._net.setInputSize((width, height))
            self._size = (width, height)

    def detect(self, image: np.ndarray) -> list[Detection]:
        height, width = image.shape[:2]
        self._ensure(width, height)

        _, raw = self._net.detect(image)
        if raw is None:
            return []

        found: list[Detection] = []
        for row in raw:
            x, y, w, h = (int(v) for v in row[:BOX_FIELDS])
            points = row[BOX_FIELDS : BOX_FIELDS + LANDMARK_FIELDS]
            found.append(
                Detection(
                    x=x, y=y, w=w, h=h,
                    confidence=float(row[SCORE_INDEX]),
                    landmarks=Landmarks(
                        right_eye=(float(points[0]), float(points[1])),
                        left_eye=(float(points[2]), float(points[3])),
                        nose=(float(points[4]), float(points[5])),
                        right_mouth=(float(points[6]), float(points[7])),
                        left_mouth=(float(points[8]), float(points[9])),
                    ),
                )
            )
        return found
