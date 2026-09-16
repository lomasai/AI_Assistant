from __future__ import annotations

from pathlib import Path

import numpy as np

from lomas_core.errors import LomasError
from lomas_core.schema import FaceConfig
from lomas_face.embedder import EMBEDDERS, normalise

INPUT_SIZE = 112
SFACE_DIM = 128
FETCH = "python tools/fetch_models.py"


@EMBEDDERS.register("sface")
class SFaceEmbedder:
    """OpenCV's SFace, the recognition model made to pair with YuNet.

    The default because it needs nothing that is not already installed: it
    runs through the same OpenCV that runs the detector, where the arcface
    embedder needs onnxruntime and a model nobody publishes in one obvious
    place. About 35 ms a face on a Pi 4, once per track.

    The crop is resized rather than aligned on the eyes. SFace is more
    accurate aligned, and doing that needs the landmarks passed through the
    embedder interface - a known next step, not an oversight.
    """

    def __init__(self, cfg: FaceConfig) -> None:
        self.cfg = cfg
        self.dim = SFACE_DIM
        self._model = None

    def _ensure(self):
        if self._model is not None:
            return self._model

        import cv2

        model = Path(self.cfg.embedder_model_path)
        if not model.exists():
            raise LomasError(
                f"face recognition model not found at {model}. Run `{FETCH}`, "
                "or use face.embedder: mock."
            )
        self._model = cv2.FaceRecognizerSF.create(str(model), "")
        return self._model

    def embed(self, face_crop: np.ndarray) -> np.ndarray:
        import cv2

        model = self._ensure()
        resized = cv2.resize(face_crop, (INPUT_SIZE, INPUT_SIZE))
        return normalise(np.asarray(model.feature(resized)).flatten())
