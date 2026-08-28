"""Five narrow jobs, each removable.

The interesting tests here are the ones about what an agent must not do: the
nudge that must never name a child's failing, the filter that must be asked
before a sound is made, and the broken agent that must not end a lesson.
"""
from __future__ import annotations

from typing import Iterator

import pytest
import yaml

from lomas_core.clock import FakeClock
from lomas_core.config import load
from lomas_core.contracts import (
    AGENT_FAILED,
    QUESTION_ANSWERED,
    QUESTION_ASKED,
    QUIZ_ANSWERED,
    QUIZ_MARKED,
    QUIZ_POSED,
    QUIZ_REQUESTED,
    ROBOT_BLOCKED,
    ROBOT_SAY,
    ROBOT_SPOKE,
    ROBOT_STATE,
    STEP_ENTERED,
    STORY_REQUESTED,
    STUDENT_DISENGAGED,
    QuestionAsked,
    QuizAnswered,
    QuizRequested,
    StoryRequested,
    StudentDisengaged,
)
from lomas_core.events import EventBus
from lomas_llm.types import Completion, Message

from app import container, seed
from app.flow.states import SessionState

ALL_AGENTS = ["tutor", "quizmaster", "narrator", "engagement", "safety"]

# Never in a nudge. A robot that tells a child she is not paying attention,
# in front of her class, is a robot the teacher switches off for good.
BANNED = ["distracted", "attention", "listening", "listen", "concentrate", "focus"]

HEADLESS = [
    "storage.backend=memory",
    "vision.pipeline.enabled=false",
    "speech.tts.engine=null",
    "speech.stt.engine=keyboard",
    "speech.wake.engine=keyboard",
    "llm.provider=offline",
    "flow.attendance_wait_seconds=1",
    "flow.answer_wait_seconds=1",
    "flow.tick_seconds=0.1",
]


class Scripted:
    """A provider that says exactly what a test needs it to say."""

    name = "scripted"

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.asked: list[list[Message]] = []

    def complete(self, messages: list[Message], **options) -> Completion:
        self.asked.append(messages)
        return Completion(text=self.reply, provider=self.name)

    def stream(self, messages: list[Message], **options) -> Iterator[str]:
        yield self.complete(messages, **options).text


def build(*extra: str):
    cfg = load("config", "debug", [*HEADLESS, *extra], use_env=False)
    system = container.build(cfg, clock=FakeClock(), bus=EventBus(cfg.runtime.event_replay_size))
    seed.demo_class(system)
    return system


@pytest.fixture
def system():
    built = build()
    yield built
    built.close()


def agent(system, name: str):
    return next(a for a in system.agents.agents if a.name == name)


def script(system, name: str, reply: str) -> Scripted:
    """Swap one agent's provider. Per agent on purpose - it is the same seam
    `agents.settings.<name>.provider` uses in production."""
    stub = Scripted(reply)
    agent(system, name).deps.llm = stub
    return stub


def seen(system, event: str) -> list:
    return [payload for _name, payload in system.bus.replay(event)]


def during(system, step_name: str, action) -> None:
    def on_step(_event, changed) -> None:
        if changed.step == step_name:
            action(system.orchestrator.ctx)

    system.bus.subscribe(STEP_ENTERED, on_step)


# --- the tutor ------------------------------------------------------------


def test_the_tutor_answers_and_addresses_the_child_who_asked(system) -> None:
    ctx = system.orchestrator.open_session()
    system.bus.publish(
        QUESTION_ASKED,
        QuestionAsked(session_id=ctx.session_id, text="why are leaves green",
                      student_id="s1", student_name="Ananya"),
    )

    answered = seen(system, QUESTION_ANSWERED)
    assert len(answered) == 1
    assert "photosynthesis" in answered[0].answer.lower()

    spoken = [u for u in seen(system, ROBOT_SAY) if u.reason == "tutor"]
    assert spoken and spoken[0].student_name == "Ananya"


def test_the_tutor_is_told_what_has_been_taught_and_no_more(system) -> None:
    stub = script(system, "tutor", "Because of chlorophyll.")
    ctx = system.orchestrator.open_session()
    system.bus.publish(
        QUESTION_ASKED,
        QuestionAsked(session_id=ctx.session_id, text="why are leaves green"),
    )

    prompt = "\n".join(m.content for m in stub.asked[0])
    lesson = system.content.load("en").lesson_for("photosynthesis")
    assert lesson.segments[-1].say not in prompt, "the tutor can see the ending"


# --- the quizmaster -------------------------------------------------------


def answer_freely(system, ctx, response: str = "it makes its own food") -> str:
    student = system.repos["student"].list_for_class(ctx.scope)[0]
    system.bus.publish(
        QUIZ_ANSWERED,
        QuizAnswered(session_id=ctx.session_id, question_id="q1", student_id=student["id"],
                     response=response, correct=None, latency_ms=1800),
    )
    return student["id"]


def test_free_text_is_marked_after_it_is_recorded(system) -> None:
    """Two writes on purpose. The class moves on the moment a child answers;
    reading what they said takes a model, and a model takes a second."""
    script(system, "quizmaster", "CORRECT")
    ctx = system.orchestrator.open_session()

    marked_step = _quiz_step(system, ctx)
    student_id = answer_freely(system, ctx)
    marked_step()

    marked = seen(system, QUIZ_MARKED)
    assert len(marked) == 1
    assert marked[0].correct is True

    recorded = system.repos["answer"].for_session(ctx.scope, ctx.session_id)
    assert [row["correct"] for row in recorded] == [1]
    assert recorded[0]["student_id"] == student_id


def test_a_wrong_answer_is_marked_wrong(system) -> None:
    script(system, "quizmaster", "WRONG")
    ctx = system.orchestrator.open_session()

    marked_step = _quiz_step(system, ctx)
    answer_freely(system, ctx, "the moon")
    marked_step()

    assert seen(system, QUIZ_MARKED)[0].correct is False
    assert system.repos["answer"].for_session(ctx.scope, ctx.session_id)[0]["correct"] == 0


def test_multiple_choice_never_reaches_a_model(system) -> None:
    """The content pack already knows. Sending it to a model costs a round
    trip and can only make it worse."""
    stub = script(system, "quizmaster", "CORRECT")
    ctx = system.orchestrator.open_session()

    marked_step = _quiz_step(system, ctx)
    student = system.repos["student"].list_for_class(ctx.scope)[0]
    system.bus.publish(
        QUIZ_ANSWERED,
        QuizAnswered(session_id=ctx.session_id, question_id="q1", student_id=student["id"],
                     response="Option two", correct=True, latency_ms=900),
    )
    marked_step()

    assert not stub.asked
    assert not seen(system, QUIZ_MARKED)


def test_the_quizmaster_writes_a_question_when_asked(system) -> None:
    script(system, "quizmaster", "What do leaves need to make food?")
    ctx = system.orchestrator.open_session()

    system.bus.publish(QUIZ_REQUESTED, QuizRequested(session_id=ctx.session_id))

    posed = seen(system, QUIZ_POSED)
    assert posed and "leaves need" in posed[0].text


# --- the narrator ---------------------------------------------------------


def test_a_story_is_bracketed_by_a_change_of_posture(system) -> None:
    """The face screen and the servos change for a story without either of
    them reading a word of it."""
    script(system, "narrator", "Once, a seed fell into a crack in a wall.")
    ctx = system.orchestrator.open_session()

    system.bus.publish(STORY_REQUESTED, StoryRequested(session_id=ctx.session_id, topic="seeds"))

    states = [p["state"] for p in seen(system, ROBOT_STATE)]
    assert states == ["storytelling", "idle"]
    assert any("seed fell" in u.text for u in seen(system, ROBOT_SAY))


# --- engagement -----------------------------------------------------------


def drift(system, ctx, student_id: str) -> None:
    system.bus.publish(
        STUDENT_DISENGAGED,
        StudentDisengaged(track_id=7, student_id=student_id, score=0.1,
                          drifting_for=8.0, at=1.0),
    )


def test_a_drifting_child_is_invited_back_by_name(system) -> None:
    ctx = system.orchestrator.open_session()
    student = system.repos["student"].list_for_class(ctx.scope)[0]

    drift(system, ctx, student["id"])

    nudges = [u for u in seen(system, ROBOT_SAY) if u.reason == "engagement"]
    assert len(nudges) == 1
    assert student["name"].split()[0] in nudges[0].text


def test_a_nudge_is_never_a_reprimand(system) -> None:
    """The hard rule. Every nudge is a question; none of them may name the
    child's failing."""
    ctx = system.orchestrator.open_session()
    student = system.repos["student"].list_for_class(ctx.scope)[0]

    drift(system, ctx, student["id"])
    said = [u.text for u in seen(system, ROBOT_SAY) if u.reason == "engagement"]

    assert said
    for line in said:
        lowered = line.lower()
        assert not [word for word in BANNED if word in lowered], line
        assert "?" in line or line.endswith("."), line


@pytest.mark.parametrize("language", ["en", "hi"])
def test_no_nudge_phrasing_in_any_language_is_a_reprimand(language: str) -> None:
    """Checked in the file, not through the code, so a translator adding a
    line trips this before a child ever hears it."""
    body = yaml.safe_load(
        open(f"config/prompts/{language}/nudge.yaml", encoding="utf-8").read()
    )
    for line in body["lines"]:
        lowered = line.lower()
        assert not [word for word in BANNED if word in lowered], line
        assert "{name}" in line, "a nudge that names nobody lands on whoever looks up"


def test_an_unrecognised_face_is_not_nudged(system) -> None:
    """You cannot invite someone by name if you do not know their name, and a
    nudge addressed to nobody lands on whoever happens to look up."""
    ctx = system.orchestrator.open_session()

    system.bus.publish(
        STUDENT_DISENGAGED,
        StudentDisengaged(track_id=7, student_id=None, score=0.1, drifting_for=8.0, at=1.0),
    )

    assert not [u for u in seen(system, ROBOT_SAY) if u.reason == "engagement"]


# --- safety ---------------------------------------------------------------


def test_a_blocked_line_is_never_spoken() -> None:
    """The filter is in the path, not listening beside it."""
    system = build("agents.safety.blocked_terms=[gunpowder]")
    try:
        ctx = system.orchestrator.open_session()
        ctx.say("First you will need gunpowder.")

        assert not system.tts.spoken, "it reached the speaker"
        assert not seen(system, ROBOT_SPOKE)

        blocked = seen(system, ROBOT_BLOCKED)
        assert len(blocked) == 1
        assert "gunpowder" in blocked[0].reason
    finally:
        system.close()


def test_a_blocked_line_still_lands_in_the_log() -> None:
    """A school will ask what the robot refused to say. Silently dropping it
    is the answer that ends a pilot."""
    system = build("agents.safety.blocked_terms=[gunpowder]")
    try:
        ctx = system.orchestrator.open_session()
        ctx.say("First you will need gunpowder.")
        system.orchestrator.close_session()

        logged = system.repos["event"].for_session(ctx.scope, ctx.session_id)
        assert [row for row in logged if row["name"] == ROBOT_BLOCKED]
    finally:
        system.close()


def test_an_ordinary_line_goes_straight_through(system) -> None:
    ctx = system.orchestrator.open_session()
    ctx.say("Leaves make food from sunlight.")

    assert system.tts.spoken
    assert seen(system, ROBOT_SPOKE)


def test_an_unusable_verdict_does_not_silence_the_robot() -> None:
    """The offline provider answers questions; it does not judge them. A
    robot that falls silent mid-lesson is a failed class."""
    system = build("agents.safety.use_model=true")
    try:
        ctx = system.orchestrator.open_session()
        ctx.say("Leaves make food from sunlight.")
        assert system.tts.spoken
    finally:
        system.close()


def test_fail_open_can_be_turned_off() -> None:
    """A school that would rather the robot said nothing than said something
    unchecked. Their call, and it is one config key."""
    system = build("agents.safety.use_model=true", "agents.safety.fail_open=false")
    try:
        ctx = system.orchestrator.open_session()
        ctx.say("Leaves make food from sunlight.")
        assert not system.tts.spoken
    finally:
        system.close()


def test_the_model_verdict_is_obeyed(system) -> None:
    blocking = build("agents.safety.use_model=true")
    try:
        script(blocking, "safety", "BLOCK")
        ctx = blocking.orchestrator.open_session()
        ctx.say("Something a child should not hear.")

        assert not blocking.tts.spoken
        assert seen(blocking, ROBOT_BLOCKED)[0].reason == "model"
    finally:
        blocking.close()


# --- every one of them is removable ---------------------------------------


@pytest.mark.parametrize("dropped", ALL_AGENTS)
def test_removing_any_agent_still_runs_a_full_class(dropped: str) -> None:
    remaining = [name for name in ALL_AGENTS if name != dropped]
    system = build(f"agents.enabled=[{','.join(remaining)}]")
    try:
        assert system.agents.names() == remaining
        assert system.orchestrator.run() is SessionState.CLOSED

        sessions = system.repos["session"].recent(system.orchestrator.scope, 1)
        assert sessions[0]["status"] == "closed"
    finally:
        system.close()


def test_a_class_runs_with_no_agents_at_all() -> None:
    system = build("agents.enabled=[]")
    try:
        assert system.agents is None
        assert system.orchestrator.run() is SessionState.CLOSED
    finally:
        system.close()


def test_without_the_tutor_a_question_simply_gets_no_answer() -> None:
    """Not a crash, not a hang. The class takes no questions and reaches the
    quiz, which is what a school with no internet budget would run."""
    system = build("agents.enabled=[safety]")
    try:
        def ask(ctx) -> None:
            system.bus.publish(
                QUESTION_ASKED,
                QuestionAsked(session_id=ctx.session_id, text="why are leaves green"),
            )

        during(system, "interaction", ask)
        assert system.orchestrator.run() is SessionState.CLOSED
        assert not seen(system, QUESTION_ANSWERED)
    finally:
        system.close()


def test_an_agent_can_be_pinned_to_its_own_provider() -> None:
    """One cheap fast model for the filter, a better one for teaching, and
    neither change touches the other."""
    system = build("agents.settings.safety.provider=offline", "llm.provider=offline")
    try:
        assert agent(system, "safety").deps.llm is not agent(system, "tutor").deps.llm
    finally:
        system.close()


def test_a_broken_agent_does_not_end_the_lesson() -> None:
    """User mode. On the bench it is re-raised, because a swallowed exception
    in debug is how you ship the bug."""
    system = build("runtime.raise_on_handler_error=false")
    try:
        broken = agent(system, "tutor")
        broken.deps.llm = _explodes()

        def ask(ctx) -> None:
            system.bus.publish(
                QUESTION_ASKED,
                QuestionAsked(session_id=ctx.session_id, text="why are leaves green"),
            )

        during(system, "interaction", ask)
        assert system.orchestrator.run() is SessionState.CLOSED

        failed = seen(system, AGENT_FAILED)
        assert failed and failed[0].agent == "tutor"
    finally:
        system.close()


def _explodes():
    class Explodes:
        name = "explodes"

        def complete(self, messages, **options):
            raise RuntimeError("no route to host")

        def stream(self, messages, **options):
            raise RuntimeError("no route to host")

    return Explodes()


def _quiz_step(system, ctx):
    """The quiz step is what writes the answer row. Agents mark what it
    recorded, so a marking test has to have it running."""
    from app.flow.step import STEPS

    step = STEPS.create("quiz", system.cfg)
    step.enter(ctx)
    return lambda: step.exit(ctx)
