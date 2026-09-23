"""Teaching as a conversation.

The Pi taught six paragraphs back to back, took questions afterwards and ran
the quiz at the end - so the first idea was ten minutes old before anybody
was asked whether they had understood it. This step says one idea and then
hands it back: a doubt invited, or a question asked about what was just
said, with the lesson held while a child is talking.
"""
from __future__ import annotations

import pytest

from lomas_core.clock import FakeClock
from lomas_core.config import load
from lomas_core.contracts import (
    LESSON_SEGMENT,
    QUESTION_ANSWERED,
    QUESTION_ASKED,
    QUIZ_ANSWERED,
    QUIZ_POSED,
    ROBOT_SAY,
    ROBOT_STATE,
    UNDERSTANDING_CHECKED,
    QuestionAnswered,
    QuestionAsked,
    QuizAnswered,
)

from app import container, seed
from app.flow.states import SessionState, StepResult
from app.flow.steps.quiz import QuizStep
from app.flow.steps.teach import Teach

HEADLESS = [
    "storage.backend=memory",
    "vision.pipeline.enabled=false",
    "speech.tts.engine=null",
    "speech.stt.engine=keyboard",
    "speech.wake.engine=keyboard",
    "llm.provider=offline",
    "flow.tick_seconds=0.05",
    "flow.attendance_wait_seconds=1",
]


def build(*extra: str):
    cfg = load("config", "debug", [*HEADLESS, *extra], use_env=False)
    system = container.build(cfg, clock=FakeClock(), bus=container.event_bus(cfg))
    seed.demo_class(system)
    return system


@pytest.fixture
def system():
    built = build()
    yield built
    built.close()


def teaching(system, *extra: str):
    """A teach step part-way through its first idea, with the clock in hand."""
    ctx = system.orchestrator.open_session()
    step = Teach(system.cfg)
    step.enter(ctx)
    return ctx, step


def run_out(system, ctx, step, ticks: int = 200) -> None:
    """Tick until the step says it is done, moving the clock on each time."""
    for _ in range(ticks):
        if step.tick(ctx, system.clock.now()) is StepResult.DONE:
            return
        system.clock.advance(1.0)
    raise AssertionError("the lesson never finished")


def seen(system, event: str) -> list:
    return [payload for _name, payload in system.bus.replay(event)]


def said(system) -> list[str]:
    return [u.text for u in seen(system, ROBOT_SAY)]


# --- the lesson is still a lesson -----------------------------------------


def test_every_idea_is_still_taught_in_order(system) -> None:
    ctx, step = teaching(system)

    run_out(system, ctx, step)
    step.exit(ctx)

    segments = seen(system, LESSON_SEGMENT)
    assert [s.index for s in segments] == list(range(len(ctx.lesson.segments)))
    assert segments[0].total == len(ctx.lesson.segments)


def test_the_class_is_asked_after_every_idea(system) -> None:
    ctx, step = teaching(system)

    run_out(system, ctx, step)
    step.exit(ctx)

    checks = seen(system, UNDERSTANDING_CHECKED)
    assert len(checks) == len(ctx.lesson.segments), "an idea went by without being checked"
    assert {c.index for c in checks} == set(range(len(ctx.lesson.segments)))


def test_a_check_is_a_question_and_then_an_invitation(system) -> None:
    """alternate: something they must answer, then something they may ask."""
    ctx, step = teaching(system)

    run_out(system, ctx, step)
    step.exit(ctx)

    kinds = [c.kind for c in seen(system, UNDERSTANDING_CHECKED)]
    assert kinds[:4] == ["question", "doubts", "question", "doubts"]


def test_how_often_to_check_is_config() -> None:
    system = build("flow.teach.check_every=3")
    try:
        ctx, step = teaching(system)
        run_out(system, ctx, step)
        step.exit(ctx)

        checks = seen(system, UNDERSTANDING_CHECKED)
        assert len(checks) == len(ctx.lesson.segments) // 3
    finally:
        system.close()


def test_a_school_can_have_doubts_only() -> None:
    system = build("flow.teach.check_style=doubts")
    try:
        ctx, step = teaching(system)
        run_out(system, ctx, step)
        step.exit(ctx)

        assert {c.kind for c in seen(system, UNDERSTANDING_CHECKED)} == {"doubts"}
        assert not seen(system, QUIZ_POSED), "a doubts-only class was given a quiz question"
    finally:
        system.close()


def test_the_lecture_is_still_available() -> None:
    """Rule four in the other direction: a school with a fixed script puts
    the three old steps back and nothing else changes."""
    system = build("flow.sequence=[attendance,greeting,lesson,interaction,quiz,wrapup]")
    try:
        assert system.orchestrator.run() is SessionState.CLOSED
        assert not seen(system, UNDERSTANDING_CHECKED)
        assert len(seen(system, LESSON_SEGMENT)) == 6
    finally:
        system.close()


# --- the class talking back ------------------------------------------------


def test_a_named_child_is_asked_and_the_turn_goes_round(system) -> None:
    ctx, step = teaching(system)

    run_out(system, ctx, step)
    step.exit(ctx)

    asked = [c.student_name for c in seen(system, UNDERSTANDING_CHECKED) if c.kind == "question"]
    assert asked[0], "nobody was asked by name"
    assert len(set(asked)) > 1, "the same child was asked every time"


def test_the_room_is_asked_when_a_school_says_no_names() -> None:
    system = build("flow.teach.name_a_child=false")
    try:
        ctx, step = teaching(system)
        run_out(system, ctx, step)
        step.exit(ctx)

        asked = [c.student_name for c in seen(system, UNDERSTANDING_CHECKED)]
        assert not any(asked)
    finally:
        system.close()


def test_the_next_idea_waits_for_a_child_who_is_asking(system) -> None:
    """The whole point. A question mid-lesson used to be answered over the
    top of the next paragraph."""
    ctx, step = teaching(system)
    step.tick(ctx, system.clock.now())          # first idea said
    before = len(seen(system, LESSON_SEGMENT))

    system.bus.publish(QUESTION_ASKED, QuestionAsked(session_id=ctx.session_id, text="why"))
    for _ in range(5):
        system.clock.advance(1.0)
        step.tick(ctx, system.clock.now())

    assert len(seen(system, LESSON_SEGMENT)) == before, "it talked over the child"

    system.bus.publish(QUESTION_ANSWERED, QuestionAnswered(
        session_id=ctx.session_id, question="why", answer="because", provider="offline"))
    system.clock.advance(30.0)
    step.tick(ctx, system.clock.now())
    step.tick(ctx, system.clock.now())

    assert len(seen(system, LESSON_SEGMENT)) > before, "the lesson never picked up again"
    step.exit(ctx)


def test_the_lesson_says_it_is_picking_up_again(system) -> None:
    ctx, step = teaching(system)
    step.tick(ctx, system.clock.now())
    system.bus.publish(QUESTION_ASKED, QuestionAsked(session_id=ctx.session_id, text="why"))
    system.bus.publish(QUESTION_ANSWERED, QuestionAnswered(
        session_id=ctx.session_id, question="why", answer="because", provider="offline"))

    system.clock.advance(30.0)
    for _ in range(3):
        step.tick(ctx, system.clock.now())
        system.clock.advance(1.0)
    step.exit(ctx)

    assert any("carry on" in line or "back to the lesson" in line or "where we" in line
               for line in said(system)), "the next idea arrived out of nowhere"


def test_a_question_nobody_answers_does_not_stop_the_lesson(system) -> None:
    """Bounded on purpose: a microphone that dies mid-turn must not leave a
    class sitting in silence for the rest of the afternoon."""
    ctx, step = teaching(system)
    step.tick(ctx, system.clock.now())
    system.bus.publish(QUESTION_ASKED, QuestionAsked(session_id=ctx.session_id, text="why"))

    system.clock.advance(system.cfg.flow.teach.answer_hold_seconds + 1.0)
    run_out(system, ctx, step)
    step.exit(ctx)

    assert len(seen(system, LESSON_SEGMENT)) == len(ctx.lesson.segments)


def test_nothing_is_said_while_the_microphone_is_open(system) -> None:
    ctx, step = teaching(system)
    step.tick(ctx, system.clock.now())
    before = len(seen(system, LESSON_SEGMENT))

    system.bus.publish(ROBOT_STATE, {"session_id": ctx.session_id, "state": "listening",
                                     "by": "microphone", "seconds": 5.0})
    system.clock.advance(60.0)
    step.tick(ctx, system.clock.now())

    assert len(seen(system, LESSON_SEGMENT)) == before, "it spoke into an open microphone"

    system.bus.publish(ROBOT_STATE, {"session_id": ctx.session_id, "state": "idle",
                                     "by": "microphone", "seconds": 0.0})
    step.tick(ctx, system.clock.now())
    step.tick(ctx, system.clock.now())

    assert len(seen(system, LESSON_SEGMENT)) > before
    step.exit(ctx)


def test_an_open_microphone_does_not_stall_the_lesson(system) -> None:
    """Hands free, the microphone is open almost all the time: it reopens
    half a second after every turn. Waiting on it without a bound is a
    lesson that never reaches its second paragraph."""
    ctx, step = teaching(system)
    system.bus.publish(ROBOT_STATE, {"session_id": ctx.session_id, "state": "listening",
                                     "by": "microphone", "seconds": 5.0})

    run_out(system, ctx, step, ticks=400)
    step.exit(ctx)

    assert len(seen(system, LESSON_SEGMENT)) == len(ctx.lesson.segments)


def test_an_answer_is_recorded_where_it_was_given(system) -> None:
    """A check answered mid-lesson is a quiz answer like any other, or the
    report would only know about the ones given at the end."""
    ctx, step = teaching(system)
    step.tick(ctx, system.clock.now())
    posed = seen(system, QUIZ_POSED)[0]
    student = system.repos["student"].list_for_class(ctx.scope)[0]

    system.bus.publish(QUIZ_ANSWERED, QuizAnswered(
        session_id=ctx.session_id, question_id=posed.question_id, student_id=student["id"],
        response="sunlight", correct=True, latency_ms=0))
    step.exit(ctx)

    answers = system.repos["answer"].for_session(ctx.scope, ctx.session_id)
    assert [a["question_ref"] for a in answers] == [posed.question_id]


def test_the_end_of_class_quiz_does_not_ask_it_all_again(system) -> None:
    ctx, step = teaching(system)
    run_out(system, ctx, step)
    step.exit(ctx)
    during_the_lesson = {p.question_id for p in seen(system, QUIZ_POSED)}

    quiz = QuizStep(system.cfg)
    quiz.enter(ctx)
    for _ in range(20):
        if quiz.tick(ctx, system.clock.now()) is StepResult.DONE:
            break
        system.clock.advance(30.0)
    quiz.exit(ctx)

    at_the_end = [p.question_id for p in seen(system, QUIZ_POSED)][len(during_the_lesson):]
    assert not (during_the_lesson & set(at_the_end)), "the class was asked the same questions twice"


def test_a_lesson_with_no_quiz_still_checks(system) -> None:
    """A written lesson whose questions did not survive the model is still
    worth teaching, and the class should still be asked whether it followed."""
    class NoQuiz:
        def __init__(self, pack):
            self._pack = pack

        def __getattr__(self, name):
            return getattr(self._pack, name)

        def quiz_for(self, _lesson_id):
            return None

    ctx, step = teaching(system)
    ctx.content = NoQuiz(ctx.content)

    run_out(system, ctx, step)
    step.exit(ctx)

    assert {c.kind for c in seen(system, UNDERSTANDING_CHECKED)} == {"doubts"}
