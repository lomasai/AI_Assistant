"""The one place tenant scope is applied to an agent's reads.

If this file passes, an agent cannot see another school. That is the whole
claim, and it is worth more than any amount of care taken elsewhere: agents
hold no repository, so this is the only door.
"""
from __future__ import annotations

from dataclasses import fields

import pytest

from lomas_core.clock import FakeClock
from lomas_core.config import load
from lomas_core.contracts import (
    LESSON_SEGMENT,
    QUESTION_ANSWERED,
    QUESTION_ASKED,
    QuestionAsked,
)
from lomas_core.errors import LomasError
from lomas_core.events import EventBus
from lomas_store import TenantScope

from app import container, seed
from app.context.assembler import ContextAssembler
from app.context.mcp_server import HISTORY, LESSON, ROSTER, SESSION, STUDENT, ContextServer

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

TUTOR = "tutor"
OTHER_ORG = "another-school-entirely"


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


@pytest.fixture
def assembler(system) -> ContextAssembler:
    return ContextAssembler(system.cfg, system.repos, system.content)


def taught_session(system) -> str:
    """A session with something in it: a lesson part-taught and a question
    asked, which is what an agent would actually be handed mid-class."""
    ctx = system.orchestrator.open_session()
    system.bus.publish(
        QUESTION_ASKED,
        QuestionAsked(session_id=ctx.session_id, text="why are leaves green",
                      student_id=system.repos["student"].list_for_class(ctx.scope)[0]["id"],
                      student_name="Ananya"),
    )
    return ctx.session_id


# --- the boundary ---------------------------------------------------------


def test_the_assembler_refuses_a_cross_org_read(system, assembler) -> None:
    """The failure this exists to prevent: one school's context handed to an
    agent running for another."""
    session_id = taught_session(system)
    intruder = TenantScope(org_id=OTHER_ORG, school_id="any", class_id="any")

    with pytest.raises(LomasError, match=OTHER_ORG):
        assembler.for_agent(TUTOR, intruder, session_id)


def test_a_session_id_from_another_org_reads_as_missing(system, assembler) -> None:
    """Not an empty context, and not somebody else's. An error, so nobody can
    build on top of a silently blank answer."""
    taught_session(system)
    scope = system.orchestrator.scope

    with pytest.raises(LomasError):
        assembler.for_agent(TUTOR, scope, "a-session-that-does-not-exist")


def test_the_context_carries_its_own_scope(system, assembler) -> None:
    session_id = taught_session(system)
    ctx = assembler.for_agent(TUTOR, system.orchestrator.scope, session_id)

    assert ctx.scope == system.orchestrator.scope
    assert ctx.session_id == session_id


# --- what an agent may see ------------------------------------------------


def test_an_agent_sees_the_lesson_but_not_the_ending(system, assembler) -> None:
    """A tutor that can read the last segment will give it away."""
    session_id = taught_session(system)
    lesson = system.content.load("en").lesson_for("photosynthesis")
    system.bus.publish(LESSON_SEGMENT, _segment(session_id, lesson, index=0))

    ctx = assembler.for_agent(TUTOR, system.orchestrator.scope, session_id)

    assert lesson.segments[0].say in ctx.lesson_text
    assert lesson.segments[-1].say not in ctx.lesson_text


def test_the_lesson_window_is_the_limit(system, assembler) -> None:
    session_id = taught_session(system)
    lesson = system.content.load("en").lesson_for("photosynthesis")
    for index in range(len(lesson.segments)):
        system.bus.publish(LESSON_SEGMENT, _segment(session_id, lesson, index))

    ctx = assembler.for_agent(TUTOR, system.orchestrator.scope, session_id)
    said = [s.say for s in lesson.segments if s.say in ctx.lesson_text]

    assert len(said) == system.cfg.context.lesson_window


def test_history_is_conversation_and_nothing_else(system, assembler) -> None:
    """Step transitions and attendance rows are machinery. An agent reasoning
    about them is an agent about to say something strange."""
    session_id = taught_session(system)
    ctx = assembler.for_agent(TUTOR, system.orchestrator.scope, session_id)

    assert ctx.history
    assert {turn.event for turn in ctx.history} <= {QUESTION_ASKED, QUESTION_ANSWERED,
                                                    "quiz.posed", "quiz.answered"}
    assert any("leaves green" in turn.text for turn in ctx.history)


def test_history_is_capped(system) -> None:
    capped = build("context.history_turns=2")
    try:
        session_id = taught_session(capped)
        for _ in range(6):
            capped.bus.publish(
                QUESTION_ASKED,
                QuestionAsked(session_id=session_id, text="and why is that"),
            )

        assembler = ContextAssembler(capped.cfg, capped.repos, capped.content)
        ctx = assembler.for_agent(TUTOR, capped.orchestrator.scope, session_id)
        assert len(ctx.history) == 2
    finally:
        capped.close()


def test_a_student_profile_is_a_name_and_a_score_not_a_file(system, assembler) -> None:
    session_id = taught_session(system)
    student = system.repos["student"].list_for_class(system.orchestrator.scope)[0]

    ctx = assembler.for_agent(TUTOR, system.orchestrator.scope, session_id, student["id"])

    assert ctx.student is not None
    assert ctx.student.name == student["name"]
    assert ctx.student_name == student["name"].split()[0]
    shown = {f.name for f in fields(ctx.student)}
    assert shown == {"student_id", "name", "roll_no", "answered", "correct"}


def test_the_profile_can_be_switched_off(system) -> None:
    """A school that will not have children profiled still gets working
    agents; they simply address the room."""
    private = build("context.include_student_profile=false")
    try:
        session_id = taught_session(private)
        student = private.repos["student"].list_for_class(private.orchestrator.scope)[0]
        assembler = ContextAssembler(private.cfg, private.repos, private.content)

        ctx = assembler.for_agent(TUTOR, private.orchestrator.scope, session_id, student["id"])
        assert ctx.student is None
    finally:
        private.close()


# --- the MCP surface ------------------------------------------------------


def test_mcp_resources_come_through_the_same_door(system, assembler) -> None:
    """Whatever MCP exposes, it exposes through the assembler - so an outside
    client is scoped by the same code that scopes an in-process agent."""
    session_id = taught_session(system)
    server = ContextServer(assembler)
    scope = system.orchestrator.scope

    assert server.enabled
    assert {r.uri for r in server.list_resources()} >= {SESSION, LESSON, HISTORY, ROSTER}

    session = server.read(SESSION, scope, session_id)
    assert session["topic"] == "photosynthesis"
    assert session["language"] == "en"

    assert "turns" in server.read(HISTORY, scope, session_id)
    assert "present" in server.read(ROSTER, scope, session_id)


def test_mcp_refuses_a_cross_org_read(system, assembler) -> None:
    session_id = taught_session(system)
    server = ContextServer(assembler)
    intruder = TenantScope(org_id=OTHER_ORG, school_id="any", class_id="any")

    with pytest.raises(LomasError):
        server.read(SESSION, intruder, session_id)


def test_an_unknown_resource_says_what_exists(system, assembler) -> None:
    session_id = taught_session(system)
    server = ContextServer(assembler)

    with pytest.raises(LomasError, match="lomas://lesson"):
        server.read("lomas://everything", system.orchestrator.scope, session_id)


def test_a_student_resource_carries_the_profile(system, assembler) -> None:
    session_id = taught_session(system)
    student = system.repos["student"].list_for_class(system.orchestrator.scope)[0]
    server = ContextServer(assembler)

    body = server.read(f"{STUDENT}/{student['id']}", system.orchestrator.scope, session_id)
    assert body["name"] == student["name"]


def _segment(session_id: str, lesson, index: int):
    from lomas_core.contracts import LessonSegment

    return LessonSegment(
        session_id=session_id,
        lesson_id=lesson.id,
        segment_id=lesson.segments[index].id,
        index=index,
        total=len(lesson.segments),
        say=lesson.segments[index].say,
        display=lesson.segments[index].display,
    )
