"""The robot listening without being asked to.

Two places a button is the wrong instrument: straight after the robot has
asked "what shall we learn today?", where it should be the one waiting, and
in the middle of a conversation, where a child should just be able to talk.
Everywhere else the teacher's press still starts the turn.
"""
from __future__ import annotations

import time

import pytest

from lomas_core.clock import FakeClock, RealClock
from lomas_core.config import load
from lomas_core.contracts import (
    QUESTION_ASKED,
    ROBOT_SAY,
    STEP_ENTERED,
    STEP_EXITED,
    TOPIC_CHOSEN,
    TOPIC_REQUESTED,
    StepChanged,
    TopicChosen,
)

from app import container, seed
from app.ears import Ears
from app.flow.states import StepResult
from app.flow.steps.topic import Topic
from app.listener import Listener

HEADLESS = [
    "storage.backend=memory",
    "vision.pipeline.enabled=false",
    "hardware.enabled=false",
    "speech.tts.engine=null",
    "speech.stt.engine=keyboard",
    "speech.wake.engine=keyboard",
    "llm.provider=offline",
    "flow.tick_seconds=0.05",
    "flow.attendance_wait_seconds=0.1",
]


def build(*extra: str, cache=""):
    # Written lessons go to a scratch directory: a cached solar system from
    # another test would make "the model refused" look like success.
    where = cache or "data/test-lessons"
    cfg = load("config", "debug", [*HEADLESS, f"content.author.cache_dir={where}", *extra],
               use_env=False)
    system = container.build(cfg, clock=FakeClock(), bus=container.event_bus(cfg))
    seed.real_class(system)
    return system


@pytest.fixture
def system(tmp_path):
    built = build(cache=tmp_path.as_posix())
    yield built
    built.close()


def hearing(system, said: str, clock=None):
    from tests.test_listener import FakeEars, FakeMic

    listener = Listener(system.cfg, system.bus, clock or system.clock, FakeMic(), FakeEars(said),
                        speakers=system.speakers)
    system.listener = listener
    return Ears(system.cfg, system.bus, clock or system.clock, listener)


def said(system, event: str) -> list:
    return [p for _n, p in system.bus.replay(event)]


# --- the topic, spoken --------------------------------------------------


def test_the_robot_asks_and_then_listens(system) -> None:
    """The greeting has asked "what shall we learn about today?" since the
    first week and nothing was listening."""
    hearing(system, "today we want to learn about the solar system")
    ctx = system.orchestrator.open_session()

    step = Topic(system.cfg)
    step.enter(ctx)

    asked = [u.text for u in said(system, ROBOT_SAY)]
    assert asked, "the class was never asked"
    assert said(system, TOPIC_REQUESTED), "asked, and nobody listening"

    deadline = time.monotonic() + 5
    while not said(system, TOPIC_CHOSEN) and time.monotonic() < deadline:
        time.sleep(0.02)

    chosen = said(system, TOPIC_CHOSEN)
    assert [c.text for c in chosen] == ["the solar system"], "the subject, not the sentence"
    assert chosen[0].by == "microphone"


def test_a_topic_the_teacher_typed_is_not_asked_about_again(system) -> None:
    hearing(system, "anything at all")
    ctx = system.orchestrator.open_session(topic="photosynthesis")

    step = Topic(system.cfg)
    step.enter(ctx)

    assert not said(system, TOPIC_REQUESTED), "the person who set it up was ignored"
    assert step.tick(ctx, system.clock.now()) is StepResult.DONE


def test_a_class_that_says_nothing_gets_the_lesson_it_had(system) -> None:
    ctx = system.orchestrator.open_session()
    step = Topic(system.cfg)
    step.enter(ctx)

    assert step.tick(ctx, system.clock.now()) is StepResult.CONTINUE
    system.clock.advance(system.cfg.flow.topic_wait_seconds + 1)

    assert step.tick(ctx, system.clock.now()) is StepResult.DONE
    step.exit(ctx)
    assert ctx.lesson.id == "photosynthesis"


def test_what_the_class_asked_for_becomes_the_lesson(system) -> None:
    from lomas_llm import Completion
    from tests.test_author import WRITTEN
    import json

    ctx = system.orchestrator.open_session()
    ctx.notes["author"].llm = type("Stub", (), {
        "complete": lambda _s, *_a, **_k: Completion(text=json.dumps(WRITTEN), provider="stub")})()

    step = Topic(system.cfg)
    step.enter(ctx)
    system.bus.publish(TOPIC_CHOSEN, TopicChosen(session_id=ctx.session_id, text="solar system"))
    step.tick(ctx, system.clock.now())
    step.exit(ctx)

    assert ctx.lesson.id == "solar-system"
    assert ctx.lesson.written is True
    assert ctx.topic == "solar-system"
    assert ctx.content.quiz_for("solar-system") is not None


def test_a_lesson_that_cannot_be_written_leaves_the_class_teaching(system) -> None:
    ctx = system.orchestrator.open_session()
    ctx.notes["author"].llm = type("Stub", (), {
        "complete": lambda _s, *_a, **_k: (_ for _ in ()).throw(RuntimeError("no"))})()

    step = Topic(system.cfg)
    step.enter(ctx)
    system.bus.publish(TOPIC_CHOSEN, TopicChosen(session_id=ctx.session_id, text="solar system"))
    step.exit(ctx)

    assert ctx.lesson.id == "photosynthesis"
    assert ctx.notes["topic_error"]


# --- a conversation with no button --------------------------------------


def test_a_child_can_just_talk_during_the_interaction(system) -> None:
    ears = hearing(system, "why do leaves fall in winter", clock=RealClock())

    system.bus.publish(STEP_ENTERED, StepChanged(session_id="s1", step="interaction", at=0.0))
    deadline = time.monotonic() + 5
    while not said(system, QUESTION_ASKED) and time.monotonic() < deadline:
        time.sleep(0.02)
    system.bus.publish(STEP_EXITED, StepChanged(session_id="s1", step="interaction", at=1.0))

    asked = said(system, QUESTION_ASKED)
    assert asked and asked[0].text == "why do leaves fall in winter"
    assert ears.turns >= 1


def test_the_robot_does_not_listen_through_the_lesson(system) -> None:
    ears = hearing(system, "anything")

    system.bus.publish(STEP_ENTERED, StepChanged(session_id="s1", step="lesson", at=0.0))
    time.sleep(0.1)

    assert not ears.listening, "it would hear its own lesson and answer itself"
    assert ears.turns == 0


def test_which_steps_it_listens_through_is_config() -> None:
    system = build("speech.audio.hands_free_steps=[quiz]")
    try:
        ears = hearing(system, "chlorophyll", clock=RealClock())
        system.bus.publish(STEP_ENTERED, StepChanged(session_id="s1", step="quiz", at=0.0))
        time.sleep(0.2)
        system.bus.publish(STEP_EXITED, StepChanged(session_id="s1", step="quiz", at=1.0))

        assert ears.turns >= 1
    finally:
        system.close()


def test_hands_free_can_be_switched_off_entirely() -> None:
    system = build("speech.audio.hands_free=false")
    try:
        ears = hearing(system, "anything")
        system.bus.publish(STEP_ENTERED, StepChanged(session_id="s1", step="interaction", at=0.0))
        time.sleep(0.1)

        assert ears.turns == 0, "press to talk, as before"
    finally:
        system.close()


def test_a_microphone_that_goes_away_mid_class_stops_the_loop(system) -> None:
    from lomas_core.errors import LomasError

    ears = hearing(system, "anything", clock=RealClock())
    system.listener.listen = lambda **_: (_ for _ in ()).throw(LomasError("unplugged"))
    ears.listener = system.listener

    system.bus.publish(STEP_ENTERED, StepChanged(session_id="s1", step="interaction", at=0.0))
    time.sleep(0.3)

    assert not ears.listening, "a lost microphone is a class driven by hand, not a spin"
