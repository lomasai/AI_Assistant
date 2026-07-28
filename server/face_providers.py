"""Local face detection and embedding providers."""

from __future__ import annotations

import base64
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from server.config import PROJECT_ROOT, RecognitionConfig


class FaceProviderError(Exception):
    """Raised when a configured local face provider cannot run safely."""


@dataclass(slots=True)
class FaceProviderHealth:
    provider: str
    ready: bool
    warmed_up: bool
    error: str | None = None

    def safe_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "ready": self.ready,
            "warmed_up": self.warmed_up,
            "error": self.error,
        }


class MockFaceEmbeddingProvider:
    """Deterministic mock embeddings for tests and development."""

    provider_name = "mock"

    def __init__(self) -> None:
        self._health = FaceProviderHealth(provider="mock", ready=True, warmed_up=True)

    async def initialize(self) -> None:
        return

    def health(self) -> dict[str, Any]:
        return self._health.safe_dict()

    def embed_seed(self, seed: str) -> list[float]:
        digest = hashlib.blake2b(seed.encode("utf-8") or b"\x00", digest_size=32).digest()
        values = [(byte - 127.5) / 127.5 for byte in digest]
        return normalize_embedding(values)

    def embed_image(self, image_base64: str) -> list[float]:
        clean = image_base64.split(",", 1)[-1].strip()
        try:
            data = base64.b64decode(clean, validate=False)
        except Exception:  # noqa: BLE001
            data = clean.encode("utf-8")
        return self.embed_seed(data.hex())


class OpenCVYuNetSFaceProvider:
    """OpenCV YuNet detector plus SFace recognizer for local Raspberry Pi use."""

    provider_name = "opencv_sface"

    def __init__(self, config: RecognitionConfig) -> None:
        self.config = config
        self.detector_model = _resolve_model_path(config.face_detection_model_path)
        self.recognizer_model = _resolve_model_path(config.face_recognition_model_path or config.embedding_model_path)
        self.detector: Any = None
        self.recognizer: Any = None
        self._health = FaceProviderHealth(provider=self.provider_name, ready=False, warmed_up=False)

    async def initialize(self) -> None:
        if not self.detector_model.exists() or not self.recognizer_model.exists():
            self._health = FaceProviderHealth(
                provider=self.provider_name,
                ready=False,
                warmed_up=False,
                error="configured face model files are unavailable",
            )
            raise FaceProviderError("Configured OpenCV face model files are unavailable.")
        try:
            import cv2  # noqa: PLC0415
        except Exception as exc:  # noqa: BLE001
            self._health = FaceProviderHealth(provider=self.provider_name, ready=False, warmed_up=False, error="opencv unavailable")
            raise FaceProviderError("OpenCV is unavailable for local face recognition.") from exc
        if not hasattr(cv2, "FaceDetectorYN_create") or not hasattr(cv2, "FaceRecognizerSF_create"):
            self._health = FaceProviderHealth(
                provider=self.provider_name,
                ready=False,
                warmed_up=False,
                error="opencv face APIs unavailable",
            )
            raise FaceProviderError("OpenCV was built without YuNet/SFace APIs.")
        self.detector = cv2.FaceDetectorYN_create(str(self.detector_model), "", (320, 320))
        self.recognizer = cv2.FaceRecognizerSF_create(str(self.recognizer_model), "")
        self._warm_up()
        self._health = FaceProviderHealth(provider=self.provider_name, ready=True, warmed_up=True)

    def health(self) -> dict[str, Any]:
        return self._health.safe_dict()

    def detect_faces(self, image_base64: str) -> list[np.ndarray]:
        image = decode_image(image_base64)
        if image is None:
            return []
        return self.detect_faces_from_image(image)

    def detect_faces_from_image(self, image: np.ndarray) -> list[np.ndarray]:
        if self.detector is None:
            raise FaceProviderError("OpenCV face detector is not initialized.")
        height, width = image.shape[:2]
        self.detector.setInputSize((width, height))
        _, faces = self.detector.detect(image)
        if faces is None:
            return []
        return [face for face in faces]

    def align_face(self, image: np.ndarray, face: np.ndarray) -> np.ndarray:
        if self.recognizer is None:
            raise FaceProviderError("OpenCV face recognizer is not initialized.")
        return self.recognizer.alignCrop(image, face)

    def embed_aligned(self, aligned_face: np.ndarray) -> list[float]:
        if self.recognizer is None:
            raise FaceProviderError("OpenCV face recognizer is not initialized.")
        feature = self.recognizer.feature(aligned_face)
        return normalize_embedding(np.asarray(feature, dtype=np.float32).flatten().tolist())

    def embed_image(self, image_base64: str) -> tuple[list[float], int]:
        image = decode_image(image_base64)
        if image is None:
            return [], 0
        faces = self.detect_faces_from_image(image)
        if len(faces) != 1:
            return [], len(faces)
        aligned = self.align_face(image, faces[0])
        return self.embed_aligned(aligned), 1

    def _warm_up(self) -> None:
        image = np.zeros((112, 112, 3), dtype=np.uint8)
        try:
            self.detect_faces_from_image(image)
        except Exception:
            pass


def build_face_embedding_provider(config: RecognitionConfig) -> MockFaceEmbeddingProvider | OpenCVYuNetSFaceProvider:
    if config.face_detection_provider == "mock" and config.face_recognition_provider == "mock":
        return MockFaceEmbeddingProvider()
    if config.face_detection_provider == "opencv" and config.face_recognition_provider == "local":
        return OpenCVYuNetSFaceProvider(config)
    raise FaceProviderError("Face detection and recognition providers must both be mock or OpenCV/local.")


def decode_image(image_base64: str) -> np.ndarray | None:
    clean = image_base64.split(",", 1)[-1].strip()
    if not clean:
        return None
    try:
        import cv2  # noqa: PLC0415

        data = base64.b64decode(clean, validate=False)
        buffer = np.frombuffer(data, dtype=np.uint8)
        return cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    except Exception:  # noqa: BLE001
        return None


def normalize_embedding(values: list[float] | np.ndarray) -> list[float]:
    vector = np.asarray(values, dtype=np.float32).flatten()
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm == 0.0:
        raise FaceProviderError("Face embedding cannot be normalized.")
    return (vector / norm).astype(float).tolist()


def _resolve_model_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path

