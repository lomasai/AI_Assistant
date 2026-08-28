from __future__ import annotations

from pathlib import Path

import numpy as np

from lomas_core.errors import LomasError
from lomas_core.schema import FaceConfig
from lomas_face.embedder import EMBEDDERS, normalise

INPUT_SIZE = 112
PIXEL_MEAN = 127.5
PIXEL_SCALE = 128.0


@EMBEDDERS.register("arcface_onnx")
class ArcFaceEmbedder:
    """ArcFace or MobileFaceNet through onnxruntime, 30-80ms on a Pi 4.

    That cost is why identity is resolved once per track rather than per
    frame - five faces at 15fps would be several cores of work.
    """

    def __init__(self, cfg: FaceConfig) -> None:
        self.cfg = cfg
        self.dim = cfg.embedding_dim
        self._session = None
        self._input_name = ""

    def _ensure(self) -> None:
        if self._session is not None:
            return
        try:
            import onnxruntime
        except ImportError as exc:
            raise LomasError(
                "onnxruntime is not installed. pip install onnxruntime, or use "
                "face.embedder: mock."
            ) from exc

        model = Path(self.cfg.embedder_model_path)
        if not model.exists():
            raise LomasError(
                f"face embedding model not found at {model}. Fetch a MobileFaceNet "
                "or ArcFace ONNX model and point face.embedder_model_path at it."
            )
        self._session = onnxruntime.InferenceSession(
            str(model), providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name

    def embed(self, face_crop: np.ndarray) -> np.ndarray:
        import cv2

        self._ensure()
        resized = cv2.resize(face_crop, (INPUT_SIZE, INPUT_SIZE))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
        batch = ((rgb - PIXEL_MEAN) / PIXEL_SCALE).transpose(2, 0, 1)[np.newaxis, :]
        output = self._session.run(None, {self._input_name: batch})[0]
        return normalise(np.asarray(output).flatten())
