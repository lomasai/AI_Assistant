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
STEP_SKIPPED = "step.skipped"

ATTENDANCE_MARKED = "attendance.marked"

# The teacher, reaching in. Every one of these is a request, never a report.
TEACHER_NUDGING = "teacher.nudging"
TEACHER_SPEAKER = "teacher.speaker"

STUDENT_ENROLLED = "student.enrolled"

ROBOT_SAY = "robot.say"
ROBOT_SPOKE = "robot.spoke"
ROBOT_STATE = "robot.state"
ROBOT_BLOCKED = "robot.blocked"

LESSON_SEGMENT = "lesson.segment"
STORY_REQUESTED = "story.requested"

QUESTION_ASKED = "question.asked"
QUESTION_ANSWERED = "question.answered"

QUIZ_REQUESTED = "quiz.requested"
QUIZ_POSED = "quiz.posed"
QUIZ_ANSWERED = "quiz.answered"
QUIZ_RECORDED = "quiz.recorded"
QUIZ_MARKED = "quiz.marked"

VISION_TRACKS = "vision.tracks"

STUDENT_IDENTIFIED = "student.identified"
STUDENT_LEFT = "student.left"
STUDENT_DISENGAGED = "student.disengaged"

AGENT_FAILED = "agent.failed"

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


@dataclass(frozen=True, slots=True)
class StoryRequested:
    session_id: str
    topic: str
    language: str = ""


@dataclass(frozen=True, slots=True)
class QuizMarked:
    """A free-text answer after the quizmaster has read it. Multiple choice
    is marked by the content pack and never reaches an agent."""

    session_id: str
    question_id: str
    student_id: str
    correct: bool | None
    comment: str = ""


@dataclass(frozen=True, slots=True)
class Blocked:
    """What the safety filter stopped. Never silently dropped: a blocked line
    is the thing a school will ask about, so it lands in the log."""

    text: str
    reason: str
    session_id: str = ""


@dataclass(frozen=True, slots=True)
class AgentFailed:
    agent: str
    event: str
    error: str
    at: float = 0.0


@dataclass(frozen=True, slots=True)
class QuizRequested:
    """The teacher asking for a question the content pack does not have."""

    session_id: str
    topic: str = ""
    count: int = 1


@dataclass(frozen=True, slots=True)
class NudgingSet:
    """A teacher silencing the robot for this session.

    Prominent on purpose. Teachers trusting they can switch it off is what
    makes them willing to leave it on.
    """

    enabled: bool
    session_id: str = ""


@dataclass(frozen=True, slots=True)
class SpeakerSet:
    """Who is talking right now, according to the person in the room.

    Voice alone will not tell you which child spoke in a classroom of forty,
    so the teacher tapping a name is the attribution.
    """

    student_id: str
    student_name: str = ""
    session_id: str = ""


@dataclass(frozen=True, slots=True)
class StudentEnrolled:
    """Vectors, angles and a count. There is no image in this payload because
    there is no image anywhere after `add_frame` returns."""

    student_id: str
    name: str
    vectors: int
    angles: tuple[str, ...] = ()
    quality: float = 0.0
