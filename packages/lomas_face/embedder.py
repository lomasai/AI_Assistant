from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from lomas_core.registry import Registry


@runtime_checkable
class FaceEmbedder(Protocol):
    dim: int

    def embed(self, face_crop: np.ndarray) -> np.ndarray: ...


EMBEDDERS: Registry[FaceEmbedder] = Registry("face embedder")


def normalise(vector: np.ndarray) -> np.ndarray:
    """L2-normalise so cosine distance is a plain dot product."""
    magnitude = float(np.linalg.norm(vector))
    if magnitude <= 0.0:
        return vector.astype(np.float32)
    return (vector / magnitude).astype(np.float32)


def distance(left: np.ndarray, right: np.ndarray) -> float:
    """Cosine distance between two normalised vectors. 0 is identical."""
    return float(1.0 - np.dot(left, right))
