from __future__ import annotations

from dataclasses import dataclass

from lomas_core.contracts import (
    LESSON_SEGMENT,
    QUESTION_ANSWERED,
    QUESTION_ASKED,
    QUIZ_ANSWERED,
    QUIZ_POSED,
)
from lomas_core.errors import LomasError
from lomas_core.schema import Config
from lomas_store import TenantScope

from app.content import ContentLibrary

# What counts as conversation. Everything else in the log is machinery, and
# an agent reasoning about step transitions is an agent about to say
# something strange.
CONVERSATION = (
    QUESTION_ASKED,
    QUESTION_ANSWERED,
    QUIZ_POSED,
    QUIZ_ANSWERED,
)

TEXT_FIELDS = ("text", "answer", "response", "say")
WHO_FIELDS = ("student_name", "student_id")
PRESENT = "present"
BLANK = ""


@dataclass(frozen=True, slots=True)
class Turn:
    event: str
    text: str
    who: str = BLANK


@dataclass(frozen=True, slots=True)
class StudentProfile:
    """What an agent may know about a child: enough to address them and to
    pitch a question, and nothing that reads like a file on them."""

    student_id: str
    name: str
    roll_no: str = BLANK
    answered: int = 0
    correct: int = 0

    @property
    def first_name(self) -> str:
        return self.name.split()[0] if self.name else BLANK


@dataclass(frozen=True, slots=True)
class AgentContext:
    agent: str
    session_id: str
    scope: TenantScope
    language: str
    grade: str
    subject: str
    vocabulary_level: str
    topic: str
    lesson_title: str
    lesson_text: str = BLANK
    history: tuple[Turn, ...] = ()
    present: tuple[str, ...] = ()
    student: StudentProfile | None = None

    @property
    def student_name(self) -> str:
        return self.student.first_name if self.student else BLANK


class ContextAssembler:
    """The one place tenant scope is applied to an agent's reads.

    Agents hold no repository and no store handle, so this is the only route
    they have to data. An agent cannot see another school even if its prompt
    asks it to, because the question never reaches a table without a scope
    attached.
    """

    def __init__(self, cfg: Config, repos: dict, content: ContentLibrary) -> None:
        self.cfg = cfg
        self.repos = repos
        self.content = content

    def for_agent(
        self,
        agent: str,
        scope: TenantScope,
        session_id: str,
        student_id: str = BLANK,
    ) -> AgentContext:
        session = self.repos["session"].get(scope, session_id)
        if session is None:
            raise LomasError(
                f"session {session_id} is not visible to org '{scope.org_id}'"
            )

        language = session["language"] or self.cfg.content.language
        topic = session["topic"] or self.cfg.content.default_topic
        lesson = self.content.load(language).lesson_for(topic)
        events = self.repos["event"].for_session(scope, session_id)

        return AgentContext(
            agent=agent,
            session_id=session_id,
            scope=scope,
            language=language,
            grade=self.cfg.content.grade,
            subject=self.cfg.content.subject,
            vocabulary_level=self.cfg.content.vocabulary_level,
            topic=topic,
            lesson_title=lesson.title,
            lesson_text=self._taught_so_far(events, lesson),
            history=self._history(events),
            present=self._present(scope, session_id),
            student=self._profile(scope, student_id),
        )

    def _taught_so_far(self, events: list[dict], lesson) -> str:
        """The segments already spoken, not the whole lesson.

        An agent that can see the ending will give it away, and a model asked
        to hold nine hundred words of context on a Pi is a model that answers
        slowly.
        """
        window = self.cfg.context.lesson_window
        if not window:
            return BLANK

        spoken = [_payload(e) for e in events if e["name"] == LESSON_SEGMENT]
        indexes = {int(p["index"]) for p in spoken if p.get("index") is not None}
        if not indexes:
            return BLANK

        taught = [s.say for i, s in enumerate(lesson.segments) if i in indexes]
        return "\n\n".join(taught[-window:])

    def _history(self, events: list[dict]) -> tuple[Turn, ...]:
        turns = [
            Turn(event=e["name"], text=_first(_payload(e), TEXT_FIELDS),
                 who=_first(_payload(e), WHO_FIELDS))
            for e in events
            if e["name"] in CONVERSATION
        ]
        kept = [t for t in turns if t.text]
        return tuple(kept[-self.cfg.context.history_turns:]) if self.cfg.context.history_turns else ()

    def _present(self, scope: TenantScope, session_id: str) -> tuple[str, ...]:
        roster = self.repos["session"].roster(scope, session_id)
        return tuple(row["name"] for row in roster if row.get("status") == PRESENT)

    def _profile(self, scope: TenantScope, student_id: str) -> StudentProfile | None:
        if not student_id or not self.cfg.context.include_student_profile:
            return None

        row = self.repos["student"].get(scope, student_id)
        if row is None:
            return None

        recent = self.repos["answer"].for_student(scope, student_id)
        recent = recent[-self.cfg.context.recent_answers:] if self.cfg.context.recent_answers else []
        return StudentProfile(
            student_id=student_id,
            name=row["name"],
            roll_no=row.get("roll_no") or BLANK,
            answered=len(recent),
            correct=sum(1 for r in recent if r["correct"]),
        )


def _payload(event: dict) -> dict:
    import json

    try:
        body = json.loads(event["payload"])
    except (TypeError, ValueError):
        return {}
    return body if isinstance(body, dict) else {}


def _first(payload: dict, fields: tuple[str, ...]) -> str:
    for field in fields:
        value = payload.get(field)
        if value:
            return str(value)
    return BLANK
