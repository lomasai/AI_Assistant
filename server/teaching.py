"""Deterministic teaching-session state machine."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from server.config import TeachingConfig
from server.interfaces import LLMProvider, SessionRepository


TeachingState = Literal[
    "idle",
    "session_setup",
    "lesson_ready",
    "explaining",
    "asking_question",
    "waiting_for_answer",
    "evaluating",
    "remediation",
    "recap",
    "session_complete",
    "paused",
    "error",
]
EvaluationLabel = Literal["correct", "partially_correct", "incorrect", "unclear"]
InterventionState = Literal[
    "possible_absence",
    "gentle_prompt",
    "question_repeat",
    "short_recap",
    "teacher_assistance_suggested",
]


ALLOWED_TRANSITIONS: dict[str, set[str]] = {
    "idle": {"session_setup"},
    "session_setup": {"lesson_ready", "paused", "session_complete", "error"},
    "lesson_ready": {"explaining", "paused", "session_complete", "error"},
    "explaining": {"asking_question", "paused", "session_complete", "error"},
    "asking_question": {"waiting_for_answer", "paused", "session_complete", "error"},
    "waiting_for_answer": {"evaluating", "paused", "session_complete", "error"},
    "evaluating": {"remediation", "recap", "session_complete", "error"},
    "remediation": {"asking_question", "waiting_for_answer", "paused", "session_complete", "error"},
    "recap": {"session_complete", "paused", "error"},
    "paused": {"session_setup", "lesson_ready", "explaining", "asking_question", "waiting_for_answer", "session_complete", "error"},
    "session_complete": set(),
    "error": {"session_complete"},
}


class TeachingError(Exception):
    """Controlled teaching workflow error."""


class InvalidTransitionError(TeachingError):
    """Raised when a command is invalid for the current state."""


class LessonConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    student_display_name: str = Field(min_length=1, max_length=80)
    grade_level: str = Field(default="middle_school", min_length=1, max_length=80)
    topic: str = Field(min_length=1, max_length=120)
    language: str = Field(default="en", min_length=1, max_length=40)
    objective: str = Field(min_length=1, max_length=300)
    duration_minutes: int | None = Field(default=None, gt=0, le=180)
    llm_provider: str | None = None


class LearningObjective(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str


class StudentResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer_text: str = Field(min_length=1, max_length=1000)


class EvaluationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: EvaluationLabel
    feedback: str
    needs_remediation: bool


class TutorOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    speech_text: str
    screen_title: str
    screen_points: list[str] = Field(default_factory=list)
    expected_response_type: Literal["spoken_or_text", "text", "spoken"] = "spoken_or_text"
    evaluation_criteria: list[str] = Field(default_factory=list)
    suggested_next_state: TeachingState = "asking_question"


class TeachingTurn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    turn_id: str
    role: Literal["tutor", "student", "system"]
    state: TeachingState
    text: str
    timestamp_utc: str
    tutor_output: TutorOutput | None = None
    evaluation: EvaluationResult | None = None


class SessionProgress(BaseModel):
    model_config = ConfigDict(extra="forbid")

    completed_turns: int = 0
    max_turns: int = 8
    remediation_attempts: int = 0
    max_remediation_attempts: int = 2


class SessionSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    student_display_name: str
    topic: str
    final_state: TeachingState
    recap: str
    evaluations: list[EvaluationResult] = Field(default_factory=list)


class TeachingSession(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    config: LessonConfig
    state: TeachingState = "session_setup"
    previous_state: TeachingState | None = None
    progress: SessionProgress
    turns: list[TeachingTurn] = Field(default_factory=list)
    current_question_id: str | None = None
    answered_question_ids: set[str] = Field(default_factory=set)
    created_at_utc: str
    updated_at_utc: str
    summary: SessionSummary | None = None


class TeachingOrchestrator:
    """Deterministic one-session teaching orchestrator."""

    def __init__(self, repository: SessionRepository, llm_provider: LLMProvider, config: TeachingConfig) -> None:
        self.repository = repository
        self.llm_provider = llm_provider
        self.config = config
        self._active_session_id: str | None = None
        self._events: dict[str, list[dict[str, Any]]] = {}

    async def create_session(self, lesson: LessonConfig) -> TeachingSession:
        now = _now()
        session = TeachingSession(
            id=str(uuid4()),
            config=lesson,
            state="session_setup",
            progress=SessionProgress(
                max_turns=self.config.max_lesson_turns,
                max_remediation_attempts=self.config.max_remediation_attempts,
            ),
            created_at_utc=now,
            updated_at_utc=now,
        )
        self._active_session_id = session.id
        await self._save(session)
        return session

    async def recover_active_sessions(self) -> list[TeachingSession]:
        if not hasattr(self.repository, "list_active_sessions"):
            return []
        raw_sessions = await self.repository.list_active_sessions()  # type: ignore[attr-defined]
        sessions = []
        for raw in raw_sessions:
            try:
                sessions.append(TeachingSession.model_validate(raw))
            except ValidationError:
                continue
        if sessions:
            self._active_session_id = sessions[0].id
            for session in sessions:
                self._events.setdefault(
                    session.id,
                    [{"type": "state", "state": session.state, "timestamp_utc": session.updated_at_utc}],
                )
        return sessions

    async def get_session(self, session_id: str) -> TeachingSession:
        raw = await self.repository.get_session(session_id)
        if raw is None:
            raise TeachingError("Teaching session not found.")
        return TeachingSession.model_validate(raw)

    async def start(self, session_id: str) -> TeachingSession:
        session = await self.get_session(session_id)
        if session.state in {"waiting_for_answer", "session_complete"}:
            return session
        self._transition(session, "lesson_ready")
        self._transition(session, "explaining")
        tutor_output = await self._tutor_output(session, "Introduce the objective and a simple example.")
        session.turns.append(_tutor_turn("explaining", tutor_output.speech_text, tutor_output))
        self._transition(session, "asking_question")
        question = TutorOutput(
            speech_text=f"Check question: what is the main idea of {session.config.topic}?",
            screen_title="Quick Check",
            screen_points=[f"Answer in your own words about {session.config.topic}."],
            expected_response_type="spoken_or_text",
            evaluation_criteria=["mentions the topic", "gives a clear idea"],
            suggested_next_state="waiting_for_answer",
        )
        question_turn = _tutor_turn("asking_question", question.speech_text, question)
        session.turns.append(question_turn)
        session.current_question_id = question_turn.turn_id
        self._transition(session, "waiting_for_answer")
        await self._save(session)
        return session

    async def submit_answer(self, session_id: str, response: StudentResponse) -> TeachingSession:
        session = await self.get_session(session_id)
        if session.state != "waiting_for_answer":
            raise InvalidTransitionError(f"Cannot submit answer while session is {session.state}.")
        if session.current_question_id and session.current_question_id in session.answered_question_ids:
            raise InvalidTransitionError("This question has already been answered.")
        self._transition(session, "evaluating")
        session.turns.append(_student_turn(response.answer_text))
        if session.current_question_id:
            session.answered_question_ids.add(session.current_question_id)
        evaluation = self._evaluate(response.answer_text)
        session.turns.append(_system_turn("evaluating", evaluation.feedback, evaluation=evaluation))
        session.progress.completed_turns += 1
        if evaluation.needs_remediation and session.progress.remediation_attempts < session.progress.max_remediation_attempts:
            session.progress.remediation_attempts += 1
            self._transition(session, "remediation")
            remediation = TutorOutput(
                speech_text=f"Let's try that another way. {session.config.topic} means connecting the idea to the example.",
                screen_title="Try Another Way",
                screen_points=["Look at the example again.", "Use one clear sentence.", "Then answer the check question."],
                expected_response_type="spoken_or_text",
                evaluation_criteria=["clear answer", "uses the lesson idea"],
                suggested_next_state="waiting_for_answer",
            )
            session.turns.append(_tutor_turn("remediation", remediation.speech_text, remediation))
            self._transition(session, "asking_question")
            retry = _tutor_turn("asking_question", f"Try again: what is the main idea of {session.config.topic}?", remediation)
            session.turns.append(retry)
            session.current_question_id = retry.turn_id
            self._transition(session, "waiting_for_answer")
        else:
            self._transition(session, "recap")
            session.summary = self._summary(session)
            session.turns.append(_tutor_turn("recap", session.summary.recap, TutorOutput(
                speech_text=session.summary.recap,
                screen_title="Recap",
                screen_points=[session.summary.recap],
                suggested_next_state="session_complete",
            )))
            self._transition(session, "session_complete")
        await self._save(session)
        return session

    async def pause(self, session_id: str) -> TeachingSession:
        session = await self.get_session(session_id)
        if session.state == "paused":
            return session
        session.previous_state = session.state
        self._transition(session, "paused")
        await self._save(session)
        return session

    async def resume(self, session_id: str) -> TeachingSession:
        session = await self.get_session(session_id)
        if session.state != "paused":
            raise InvalidTransitionError(f"Cannot resume while session is {session.state}.")
        target = session.previous_state or "waiting_for_answer"
        self._transition(session, target)
        session.previous_state = None
        await self._save(session)
        return session

    async def stop(self, session_id: str) -> TeachingSession:
        session = await self.get_session(session_id)
        if session.state == "session_complete":
            return session
        self._transition(session, "session_complete")
        if session.summary is None:
            session.summary = self._summary(session)
        await self._save(session)
        return session

    async def summary(self, session_id: str) -> SessionSummary:
        session = await self.get_session(session_id)
        if session.summary is None:
            session.summary = self._summary(session)
            await self._save(session)
        return session.summary

    async def record_intervention(self, session_id: str, intervention: InterventionState | str, message: str) -> TeachingSession:
        session = await self.get_session(session_id)
        if session.state in {"paused", "session_complete", "error"}:
            return session
        safe_message = message.strip()[:300] or "Ready when you are."
        session.turns.append(_system_turn(session.state, f"Engagement support: {intervention}: {safe_message}"))
        session.updated_at_utc = _now()
        self._events.setdefault(session.id, []).append(
            {
                "type": "engagement_support",
                "state": intervention,
                "message": safe_message,
                "timestamp_utc": session.updated_at_utc,
            }
        )
        await self._save(session)
        return session

    def events(self, session_id: str) -> list[dict[str, Any]]:
        return list(self._events.get(session_id, []))

    def _transition(self, session: TeachingSession, target: TeachingState) -> None:
        if target == session.state:
            return
        if target not in ALLOWED_TRANSITIONS.get(session.state, set()):
            raise InvalidTransitionError(f"Invalid transition from {session.state} to {target}.")
        session.state = target
        session.updated_at_utc = _now()
        self._events.setdefault(session.id, []).append({"type": "state", "state": target, "timestamp_utc": session.updated_at_utc})

    async def _tutor_output(self, session: TeachingSession, instruction: str) -> TutorOutput:
        prompt = (
            "teaching_structured_output\n"
            f"Topic: {session.config.topic}\nObjective: {session.config.objective}\nInstruction: {instruction}"
        )
        for _ in range(self.config.structured_output_retries + 1):
            try:
                raw = await asyncio.wait_for(
                    self.llm_provider.generate(prompt=prompt, system_prompt="Return strict JSON for a tutor turn."),
                    timeout=self.config.provider_timeout_seconds,
                )
                return TutorOutput.model_validate(json.loads(raw))
            except (TimeoutError, json.JSONDecodeError, ValidationError, Exception):
                continue
        return TutorOutput(
            speech_text=f"Today we will learn {session.config.topic}. {session.config.objective}",
            screen_title=session.config.topic,
            screen_points=[session.config.objective, "We will use one example, then a quick question."],
            suggested_next_state="asking_question",
        )

    def _evaluate(self, answer: str) -> EvaluationResult:
        text = " ".join(answer.lower().split())
        if len(text) < 3 or text in {"?", "idk", "i don't know", "dont know"}:
            return EvaluationResult(label="unclear", feedback="I could not understand the answer clearly. Try one complete sentence.", needs_remediation=True)
        if any(word in text for word in {"partly", "maybe", "some"}):
            return EvaluationResult(label="partially_correct", feedback="That is partly right. Add the main idea more clearly.", needs_remediation=True)
        if any(word in text for word in {"wrong", "incorrect", "no idea", "not sure"}):
            return EvaluationResult(label="incorrect", feedback="That is not quite right yet. Let's review the example.", needs_remediation=True)
        return EvaluationResult(label="correct", feedback="Good answer. You used the lesson idea clearly.", needs_remediation=False)

    def _summary(self, session: TeachingSession) -> SessionSummary:
        evaluations = [turn.evaluation for turn in session.turns if turn.evaluation is not None]
        last = evaluations[-1].feedback if evaluations else "We introduced the lesson objective."
        return SessionSummary(
            session_id=session.id,
            student_display_name=session.config.student_display_name,
            topic=session.config.topic,
            final_state=session.state,
            recap=f"Recap for {session.config.topic}: {session.config.objective} {last}",
            evaluations=evaluations,
        )

    async def _save(self, session: TeachingSession) -> None:
        await self.repository.save_session(session.model_dump(mode="json"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tutor_turn(state: TeachingState, text: str, tutor_output: TutorOutput) -> TeachingTurn:
    return TeachingTurn(turn_id=str(uuid4()), role="tutor", state=state, text=text, timestamp_utc=_now(), tutor_output=tutor_output)


def _student_turn(text: str) -> TeachingTurn:
    return TeachingTurn(turn_id=str(uuid4()), role="student", state="waiting_for_answer", text=text, timestamp_utc=_now())


def _system_turn(state: TeachingState, text: str, evaluation: EvaluationResult | None = None) -> TeachingTurn:
    return TeachingTurn(turn_id=str(uuid4()), role="system", state=state, text=text, timestamp_utc=_now(), evaluation=evaluation)
