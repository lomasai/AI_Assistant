from __future__ import annotations

import numpy as np

from lomas_core.schema import FaceConfig
from lomas_face.embedder import FaceEmbedder, distance
from lomas_face.quality import crop_face
from lomas_face.types import Track



class IdentityMatcher:
    """Puts a name on a track, as rarely as possible.

    Identity does not change between frames, so the embedder runs when a track
    is new and then occasionally to catch a mistake. Everything else is the
    tracker carrying the answer forward for free. This is the difference
    between the system running on a Pi 4 and not.
    """

    def __init__(self, embedder: FaceEmbedder, cfg: FaceConfig) -> None:
        self.embedder = embedder
        self.cfg = cfg
        self._enrolled: dict[str, list[np.ndarray]] = {}
        self.embed_calls = 0
        self.matches = 0
        self.unknowns = 0
        self.skipped_too_small = 0

    def load(self, enrolled: dict[str, list[np.ndarray]]) -> None:
        self._enrolled = {
            student_id: [np.asarray(v, dtype=np.float32) for v in vectors]
            for student_id, vectors in enrolled.items()
        }

    @property
    def enrolled_count(self) -> int:
        return len(self._enrolled)

    def resolve(self, track: Track, frame: np.ndarray, ts: float) -> str | None:
        if not self._should_embed(track, ts):
            return track.student_id

        if track.box.w < self.cfg.recognition_min_face_px:
            self.skipped_too_small += 1
            return track.student_id

        crop = crop_face(frame, track.box, self.cfg.crop_margin)
        if crop.size == 0:
            return track.student_id

        self.embed_calls += 1
        vector = self.embedder.embed(crop)
        track.verified_at = ts

        student_id = self._nearest(vector)
        if student_id is None:
            track.identify_attempts += 1
            self.unknowns += 1
            return track.student_id

        self.matches += 1
        track.student_id = student_id
        track.identify_attempts = 0
        return student_id

    def stats(self) -> dict[str, int]:
        return {
            "embed_calls": self.embed_calls,
            "matches": self.matches,
            "unknowns": self.unknowns,
            "skipped_too_small": self.skipped_too_small,
            "enrolled": self.enrolled_count,
        }

    def reset(self) -> None:
        self.embed_calls = 0
        self.matches = 0
        self.unknowns = 0
        self.skipped_too_small = 0

    def _should_embed(self, track: Track, ts: float) -> bool:
        # A stranger is not worth re-checking forever; after a few failures
        # this track stops costing anything at all.
        if track.student_id is None and track.identify_attempts >= self.cfg.unknown_after_attempts:
            return False

        if track.verified_at is None:
            return True

        return (ts - track.verified_at) >= self.cfg.reverify_seconds

    def _nearest(self, vector: np.ndarray) -> str | None:
        best_id: str | None = None
        best_distance = self.cfg.match_threshold

        for student_id, vectors in self._enrolled.items():
            for enrolled in vectors:
                gap = distance(vector, enrolled)
                if gap < best_distance:
                    best_distance = gap
                    best_id = student_id
        return best_id
