from __future__ import annotations

from dataclasses import dataclass, field

# Event names are the stable API between features. Adding one is free;
# changing a payload is a breaking change and should be treated as one.
#
# Nothing here imports anything. Every feature publishes and subscribes
# through these names, which is what lets any one of them be deleted without
# the rest noticing.

SESSION_OPENED = "session.opened"
SESSION_CLOSED = "session.closed"
SESSION_PAUSED = "session.paused"
SESSION_RESUMED = "session.resumed"

STEP_ENTERED = "step.entered"
STEP_EXITED = "step.exited"

ATTENDANCE_MARKED = "attendance.marked"

ROBOT_SAY = "robot.say"
ROBOT_SPOKE = "robot.spoke"
ROBOT_STATE = "robot.state"

LESSON_SEGMENT = "lesson.segment"

QUESTION_ASKED = "question.asked"
QUESTION_ANSWERED = "question.answered"

QUIZ_POSED = "quiz.posed"
QUIZ_ANSWERED = "quiz.answered"

VISION_TRACKS = "vision.tracks"

STUDENT_IDENTIFIED = "student.identified"
STUDENT_LEFT = "student.left"
STUDENT_DISENGAGED = "student.disengaged"

SAFETY_HALT = "safety.halt"
SAFETY_CLEARED = "safety.cleared"


@dataclass(frozen=True, slots=True)
class SessionOpened:
    session_id: str
    org_id: str
    school_id: str
    class_id: str
    language: str
    topic: str
    started_at: float


@dataclass(frozen=True, slots=True)
class SessionClosed:
    session_id: str
    ended_at: float
    reason: str


@dataclass(frozen=True, slots=True)
class StepChanged:
    session_id: str
    step: str
    at: float


@dataclass(frozen=True, slots=True)
class AttendanceMarked:
    session_id: str
    student_id: str
    name: str
    source: str  # recognised | roster | teacher


@dataclass(frozen=True, slots=True)
class Utterance:
    """What the robot is about to say, or has just said.

    `student_name` is set only when it is addressing one child directly. The
    greeting deliberately leaves it empty - a roll of names at the door
    singles a few out and leaves the rest waiting.
    """

    text: str
    language: str
    session_id: str = ""
    student_name: str = ""
    reason: str = ""


@dataclass(frozen=True, slots=True)
class LessonSegment:
    session_id: str
    lesson_id: str
    segment_id: str
    index: int
    total: int
    say: str
    display: str


@dataclass(frozen=True, slots=True)
class QuestionAsked:
    session_id: str
    text: str
    student_id: str = ""
    student_name: str = ""


@dataclass(frozen=True, slots=True)
class QuestionAnswered:
    session_id: str
    question: str
    answer: str
    provider: str
    student_id: str = ""


@dataclass(frozen=True, slots=True)
class QuizPosed:
    session_id: str
    question_id: str
    text: str
    options: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class QuizAnswered:
    session_id: str
    question_id: str
    student_id: str
    response: str
    correct: bool | None
    latency_ms: int


@dataclass(frozen=True, slots=True)
class SafetyHalt:
    reason: str
    at: float
    detail: dict = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TrackView:
    """One face as everything downstream sees it.

    Boxes are full-resolution pixels whatever the detector actually ran on,
    so an overlay can draw them without knowing anything about downscaling.
    """

    track_id: int
    x: int
    y: int
    w: int
    h: int
    student_id: str | None
    attention: float
    yaw: float
    pitch: float
    seen_for: float


@dataclass(frozen=True, slots=True)
class TracksSeen:
    source_id: str
    zone: str
    at: float
    width: int
    height: int
    tracks: tuple[TrackView, ...] = ()


@dataclass(frozen=True, slots=True)
class StudentIdentified:
    student_id: str
    track_id: int
    source_id: str
    zone: str
    at: float


@dataclass(frozen=True, slots=True)
class StudentLeft:
    student_id: str
    track_id: int
    seen_for: float
    at: float


@dataclass(frozen=True, slots=True)
class StudentDisengaged:
    """Deliberately not lomas_face.Disengagement. Core carries no dependency
    on a package, and this is the shape the boundary promises."""

    track_id: int
    student_id: str | None
    score: float
    drifting_for: float
    at: float
