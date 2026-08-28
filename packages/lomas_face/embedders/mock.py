from __future__ import annotations

import numpy as np

from lomas_core.schema import FaceConfig
from lomas_face.embedder import EMBEDDERS, normalise

SEED_SCALE = 1000


@EMBEDDERS.register("mock")
class MockEmbedder:
    """Deterministic vectors, no model.

    The seed is the rounded mean of the crop, so a test can represent a person
    by filling a crop with their number: the same person always embeds to the
    same vector, and different people land nearly orthogonal to each other.
    """

    def __init__(self, cfg: FaceConfig | None = None) -> None:
        self.dim = cfg.embedding_dim if cfg else 512
        self.calls = 0

    def embed(self, face_crop: np.ndarray) -> np.ndarray:
        self.calls += 1
        seed = int(round(float(face_crop.mean()))) % SEED_SCALE
        rng = np.random.default_rng(seed)
        return normalise(rng.normal(size=self.dim))
