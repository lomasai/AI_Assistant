"""A whole class, end to end, in milliseconds.

This is the test the original question was about: does the flow actually run?
It uses the fake clock, mock speech and the offline provider, so it needs no
hardware, no key and no waiting.
"""
from __future__ import annotations

import pytest

from lomas_core.clock import FakeClock
from lomas_core.config import load
from lomas_core.contracts import (
    ATTENDANCE_MARKED,
    LESSON_SEGMENT,
    QUESTION_ASKED,
    QUESTION_ANSWERED,
    QUIZ_ANSWERED,
    QUIZ_POSED,
    ROBOT_SAY,
    SAFETY_CLEARED,
    SESSION_CLOSED,
    SESSION_OPENED,
    STEP_ENTERED,
    QuestionAsked,
    QuizAnswered,
)
from lomas_core.events import EventBus

from app import container, seed
from app.flow.states import SessionState

HEADLESS = [
    "storage.backend=memory",
    "speech.tts.engine=null",
    "speech.stt.engine=keyboard",
    "speech.wake.engine=keyboard",
    "llm.provider=offline",
    "flow.attendance_wait_seconds=1",
    "flow.answer_wait_seconds=1",
    "flow.tick_seconds=0.1",
]


def build(*extra: str):
    cfg = load("config", "debug", [*HEADLESS, *extra], use_env=False)
    clock = FakeClock()
    bus = EventBus(replay_size=cfg.runtime.event_replay_size)
    system = container.build(cfg, clock=clock, bus=bus)
    seed.demo_class(system)
    return system


@pytest.fixture
def system():
    built = build()
    yield built
    built.close()


def names_of(bus: EventBus, event: str) -> list:
    return [payload for name, payload in bus.replay(event)]


# --- the whole class -------------------------------------------------------


def test_a_full_class_runs_end_to_end(system):
    state = system.orchestrator.run()

    assert state is SessionState.CLOSED
    steps = [p.step for p in names_of(system.bus, STEP_ENTERED)]
    assert steps == ["attendance", "greeting", "lesson", "interaction", "quiz", "wrapup"]

    assert names_of(system.bus, SESSION_OPENED)
    assert names_of(system.bus, SESSION_CLOSED)[0].reason == "closed"


def test_the_class_is_taught_in_order(system):
    system.orchestrator.run()
    segments = names_of(system.bus, LESSON_SEGMENT)

    assert len(segments) == 6
    assert [s.index for s in segments] == list(range(6))
    assert segments[0].total == 6
    assert "photosynthesis" in segments[0].lesson_id


def test_everything_spoken_is_recorded(system):
    system.orchestrator.run()
    spoken = [u.text for u in names_of(system.bus, ROBOT_SAY)]

    assert spoken, "the robot said nothing"
    assert any("chlorophyll" in line for line in spoken), "the lesson was not taught"
    assert system.tts.spoken, "the voice service never reached the engine"


def test_the_greeting_names_nobody(system):
    """A roll of names at the door singles a few children out. Names are for
    when the robot asks someone a question directly."""
    system.orchestrator.run()
    greeting = names_of(system.bus, ROBOT_SAY)[0]

    roster = {s["name"].split()[0] for s in system.orchestrator.repos["student"].list_for_class(
        system.orchestrator.scope
    )}
    assert not any(first in greeting.text for first in roster)
    assert greeting.student_name == ""


def test_attendance_falls_back_to_the_roster(system):
    system.orchestrator.run()
    marked = names_of(system.bus, ATTENDANCE_MARKED)

    assert len(marked) == 5
    assert {m.source for m in marked} == {"roster"}


def test_every_quiz_question_is_asked(system):
    """Nobody answers, and the class still reaches the end of the quiz."""
    system.orchestrator.run()
    posed = names_of(system.bus, QUIZ_POSED)

    assert len(posed) == 6
    assert len({p.question_id for p in posed}) == 6


def test_the_session_is_written_to_the_database(system):
    system.orchestrator.run()
    scope = system.orchestrator.scope

    sessions = system.repos["session"].recent(scope, 5)
    assert len(sessions) == 1
    assert sessions[0]["status"] == "closed"
    assert sessions[0]["ended_at"] is not None

    roster = system.repos["session"].roster(scope, sessions[0]["id"])
    assert len(roster) == 5

    logged = system.repos["event"].for_session(scope, sessions[0]["id"])
    assert len(logged) > 10, "the append-only log is what every report reads"


# --- interaction and quiz with people in the room --------------------------


def during(system, step_name: str, action):
    """Steps only listen while they are running, which is correct - questions
    are answered during question time. So inject when that step opens."""

    def on_step(_event, changed) -> None:
        if changed.step == step_name:
            action(system.orchestrator.ctx)

    system.bus.subscribe(STEP_ENTERED, on_step)


def test_a_student_question_gets_an_answer(system):
    def ask(ctx) -> None:
        system.bus.publish(
            QUESTION_ASKED,
            QuestionAsked(session_id=ctx.session_id, text="why are leaves green",
                          student_id="s1", student_name="Ananya"),
        )

    during(system, "interaction", ask)
    system.orchestrator.run()

    answered = names_of(system.bus, QUESTION_ANSWERED)
    assert len(answered) == 1
    assert answered[0].provider == "offline"
    assert "photosynthesis" in answered[0].answer.lower()
    assert answered[0].student_id == "s1"


def test_the_answer_is_addressed_to_the_child_who_asked(system):
    """Names are used here, where being named means the robot saw you."""
    def ask(ctx) -> None:
        system.bus.publish(
            QUESTION_ASKED,
            QuestionAsked(session_id=ctx.session_id, text="why are leaves green",
                          student_id="s1", student_name="Ananya"),
        )

    during(system, "interaction", ask)
    system.orchestrator.run()

    addressed = [u for u in names_of(system.bus, ROBOT_SAY) if u.student_name]
    assert addressed and addressed[0].student_name == "Ananya"


def test_quiz_answers_are_recorded_per_student(system):
    def answer(ctx) -> None:
        students = system.repos["student"].list_for_class(ctx.scope)
        for index, student in enumerate(students[:3]):
            system.bus.publish(
                QUIZ_ANSWERED,
                QuizAnswered(session_id=ctx.session_id, question_id=f"q{index + 1}",
                             student_id=student["id"], response="It makes its own",
                             correct=True, latency_ms=2400),
            )

    during(system, "quiz", answer)
    system.orchestrator.run()

    scope = system.orchestrator.scope
    session_id = system.repos["session"].recent(scope, 1)[0]["id"]
    recorded = system.repos["answer"].for_session(scope, session_id)

    assert len(recorded) == 3
    assert len({r["student_id"] for r in recorded}) == 3, "scores are per student"
    assert all(r["correct"] == 1 for r in recorded)


# --- the flow is configuration ---------------------------------------------


@pytest.mark.parametrize(
    "dropped",
    ["attendance", "greeting", "lesson", "interaction", "quiz", "wrapup"],
)
def test_removing_any_step_still_produces_a_session(dropped):
    """Rule four: the system must run with any feature switched off."""
    remaining = [s for s in
                 ["attendance", "greeting", "lesson", "interaction", "quiz", "wrapup"]
                 if s != dropped]
    system = build(f"flow.sequence=[{','.join(remaining)}]")
    try:
        assert system.orchestrator.run() is SessionState.CLOSED
        assert [p.step for p in names_of(system.bus, STEP_ENTERED)] == remaining

        sessions = system.repos["session"].recent(system.orchestrator.scope, 5)
        assert sessions[0]["status"] == "closed"
    finally:
        system.close()


def test_the_stages_can_be_reordered():
    system = build("flow.sequence=[greeting,lesson,wrapup]")
    try:
        system.orchestrator.run()
        assert [p.step for p in names_of(system.bus, STEP_ENTERED)] == [
            "greeting", "lesson", "wrapup"
        ]
    finally:
        system.close()


def test_a_different_lesson_is_a_config_change():
    system = build("content.default_topic=water-cycle")
    try:
        system.orchestrator.run()
        segments = names_of(system.bus, LESSON_SEGMENT)
        assert segments[0].lesson_id == "water-cycle"
        assert len(segments) == 5
    finally:
        system.close()


def test_an_unknown_lesson_lists_what_exists():
    system = build("content.default_topic=quantum-mechanics")
    try:
        with pytest.raises(Exception, match="photosynthesis"):
            system.orchestrator.open_session()
    finally:
        system.close()


# --- teacher control -------------------------------------------------------


def test_the_teacher_can_pause_and_resume_mid_lesson(system):
    machine = system.extras["machine"]
    seen: list[str] = []

    def on_lesson(_event, changed) -> None:
        if changed.step != "lesson":
            return
        system.orchestrator.pause()
        seen.append(machine.state.value)
        system.orchestrator.resume()
        seen.append(machine.state.value)

    system.bus.subscribe(STEP_ENTERED, on_lesson)
    assert system.orchestrator.run() is SessionState.CLOSED
    assert seen == ["paused", "running"]


def test_a_halt_raised_mid_lesson_ends_the_class(system):
    def on_lesson(_event, changed) -> None:
        if changed.step == "lesson":
            system.orchestrator.halt("emergency stop")

    system.bus.subscribe(STEP_ENTERED, on_lesson)

    assert system.orchestrator.run() is SessionState.HALTED
    assert names_of(system.bus, SESSION_CLOSED)[-1].reason == "halted"

    steps = [p.step for p in names_of(system.bus, STEP_ENTERED)]
    assert "quiz" not in steps, "the class continued after a halt"


def test_a_halt_latches_until_it_is_cleared(system):
    """Like the physical e-stop it comes from. A halt raised before a class
    begins must not be discarded by starting one."""
    system.orchestrator.halt("emergency stop")
    assert system.orchestrator.run() is SessionState.HALTED
    assert names_of(system.bus, STEP_ENTERED) == [], "it started anyway"

    system.bus.publish(SAFETY_CLEARED, {"reason": "reset"})
    assert system.orchestrator.run() is SessionState.CLOSED


def test_debug_mode_writes_to_the_scratch_tenant(system):
    assert system.cfg.active_org_id == system.cfg.tenancy.scratch_org_id
    assert system.orchestrator.scope.org_id == "scratch"


def test_seeding_twice_changes_nothing(system):
    scope = system.orchestrator.scope
    before = len(system.repos["student"].list_for_class(scope))
    seed.demo_class(system)
    assert len(system.repos["student"].list_for_class(scope)) == before
