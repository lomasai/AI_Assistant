from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from lomas_core.errors import LomasError
from lomas_core.schema import EnrolmentConfig, PoseConfig
from lomas_face.embedder import FaceEmbedder, normalise
from lomas_face.quality import assess, crop_face
from lomas_face.types import Detection

ACCEPTED = "kept"
TOO_BLURRED = "hold still, it is too blurred"
TOO_SMALL = "come a little closer"
ENOUGH_OF_THAT_ANGLE = "turn your head a bit further"


@dataclass(frozen=True, slots=True)
class FrameFeedback:
    """What the enrolment screen shows while the head turns."""

    accepted: bool
    reason: str
    quality: float
    angle: str
    collected: int
    needed: int
    angles_covered: tuple[str, ...]


@dataclass(slots=True)
class EnrolmentResult:
    mean_vector: np.ndarray
    variants: list[np.ndarray] = field(default_factory=list)
    angles: list[str] = field(default_factory=list)
    quality: float = 0.0

    @property
    def dim(self) -> int:
        return int(self.mean_vector.shape[0])

    def as_rows(self) -> list[tuple[bytes, int, str, float, str]]:
        """Ready for EmbeddingRepo.add - bytes, never pictures."""
        rows = [(self.mean_vector.tobytes(), self.dim, "float32", self.quality, "mean")]
        rows.extend(
            (vector.tobytes(), int(vector.shape[0]), "float32", self.quality, angle)
            for vector, angle in zip(self.variants, self.angles)
        )
        return rows


class EnrolmentSession:
    """Turns a few seconds of a child turning their head into vectors.

    Frames are embedded and dropped inside `add_frame`. Nothing here stores,
    caches or returns an image, which is what makes the privacy promise a
    property of the code rather than a policy someone has to remember.
    """

    def __init__(self, embedder: FaceEmbedder, cfg: EnrolmentConfig, pose_cfg: PoseConfig) -> None:
        self.embedder = embedder
        self.cfg = cfg
        self.pose_cfg = pose_cfg
        self._by_angle: dict[str, list[tuple[float, np.ndarray]]] = {}
        self.seen = 0

    @property
    def collected(self) -> int:
        return sum(len(v) for v in self._by_angle.values())

    @property
    def angles_covered(self) -> tuple[str, ...]:
        return tuple(sorted(self._by_angle))

    @property
    def complete(self) -> bool:
        return set(self.cfg.required_angles) <= set(self._by_angle)

    def add_frame(self, image: np.ndarray, detection: Detection) -> FrameFeedback:
        self.seen += 1
        quality = assess(image, detection, self.cfg, self.pose_cfg)

        if detection.w < self.cfg.min_face_px:
            return self._feedback(False, TOO_SMALL, quality.score, quality.angle)
        if quality.score < self.cfg.min_quality:
            return self._feedback(False, TOO_BLURRED, quality.score, quality.angle)

        bucket = self._by_angle.setdefault(quality.angle, [])
        if len(bucket) >= self.cfg.keep_best and quality.score <= min(s for s, _ in bucket):
            return self._feedback(False, ENOUGH_OF_THAT_ANGLE, quality.score, quality.angle)

        crop = crop_face(image, detection, self.cfg.crop_margin)
        bucket.append((quality.score, self.embedder.embed(crop)))
        bucket.sort(key=lambda pair: pair[0], reverse=True)
        del bucket[self.cfg.keep_best :]

        return self._feedback(True, ACCEPTED, quality.score, quality.angle)

    def finish(self) -> EnrolmentResult:
        samples = [pair for bucket in self._by_angle.values() for pair in bucket]
        if not samples:
            raise LomasError("enrolment captured no usable frames")

        best_per_angle = {
            angle: max(bucket, key=lambda pair: pair[0]) for angle, bucket in self._by_angle.items()
        }
        angles = sorted(best_per_angle)
        variants = [best_per_angle[a][1] for a in angles]

        mean = normalise(np.mean([vector for _, vector in samples], axis=0))
        quality = float(np.mean([score for score, _ in samples]))
        return EnrolmentResult(mean_vector=mean, variants=variants, angles=angles, quality=quality)

    def _feedback(self, accepted: bool, reason: str, quality: float, angle: str) -> FrameFeedback:
        return FrameFeedback(
            accepted=accepted,
            reason=reason,
            quality=quality,
            angle=angle,
            collected=self.collected,
            needed=len(self.cfg.required_angles) * self.cfg.keep_best,
            angles_covered=self.angles_covered,
        )
