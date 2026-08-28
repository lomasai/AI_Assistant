from __future__ import annotations

from dataclasses import dataclass, field

Point = tuple[float, float]
HALF = 2


@dataclass(frozen=True, slots=True)
class Landmarks:
    """The five points every practical face detector gives you for free."""

    right_eye: Point
    left_eye: Point
    nose: Point
    right_mouth: Point
    left_mouth: Point

    @property
    def eye_centre(self) -> Point:
        return (
            (self.right_eye[0] + self.left_eye[0]) / HALF,
            (self.right_eye[1] + self.left_eye[1]) / HALF,
        )

    @property
    def mouth_centre(self) -> Point:
        return (
            (self.right_mouth[0] + self.left_mouth[0]) / HALF,
            (self.right_mouth[1] + self.left_mouth[1]) / HALF,
        )


@dataclass(frozen=True, slots=True)
class Detection:
    x: int
    y: int
    w: int
    h: int
    confidence: float
    landmarks: Landmarks | None = None

    @property
    def box(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.x + self.w, self.y + self.h

    @property
    def centre(self) -> Point:
        return self.x + self.w / HALF, self.y + self.h / HALF

    @property
    def area(self) -> int:
        return self.w * self.h

    def scaled(self, factor: float) -> Detection:
        """Map a detection made on a downscaled copy back to full resolution."""
        return Detection(
            x=int(self.x * factor),
            y=int(self.y * factor),
            w=int(self.w * factor),
            h=int(self.h * factor),
            confidence=self.confidence,
            landmarks=None
            if self.landmarks is None
            else Landmarks(
                right_eye=(self.landmarks.right_eye[0] * factor, self.landmarks.right_eye[1] * factor),
                left_eye=(self.landmarks.left_eye[0] * factor, self.landmarks.left_eye[1] * factor),
                nose=(self.landmarks.nose[0] * factor, self.landmarks.nose[1] * factor),
                right_mouth=(
                    self.landmarks.right_mouth[0] * factor,
                    self.landmarks.right_mouth[1] * factor,
                ),
                left_mouth=(
                    self.landmarks.left_mouth[0] * factor,
                    self.landmarks.left_mouth[1] * factor,
                ),
            ),
        )


@dataclass(frozen=True, slots=True)
class Pose:
    yaw: float
    pitch: float
    roll: float


@dataclass(slots=True)
class Track:
    """A face followed across frames.

    `student_id` is filled in by the identity matcher when the track is born
    and then rides along for free, which is what keeps recognition off the
    per-frame path.
    """

    track_id: int
    box: Detection
    first_seen: float
    last_seen: float
    hits: int = 1
    misses: int = 0
    confirmed: bool = False
    student_id: str | None = None
    pose: Pose | None = None
    attention: float = 1.0
    verified_at: float | None = None  # None means never, which 0.0 cannot mean
    identify_attempts: int = 0
    history: list[Point] = field(default_factory=list)

    @property
    def age(self) -> float:
        return self.last_seen - self.first_seen


@dataclass(frozen=True, slots=True)
class Disengagement:
    track_id: int
    student_id: str | None
    score: float
    drifting_for: float
    at: float
