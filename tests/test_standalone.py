"""A robot that needs no laptop.

Three things kept one attached: only the teacher's page could start a class,
only that page could enrol a child, and both of those are screens. The robot
has a camera that notices a stranger, a voice that can ask a question and
ears that hear the answer - so it can do its own introductions and wait to
be told to begin.
"""
from __future__ import annotations

import time

import pytest

from lomas_core.clock import FakeClock, RealClock
from lomas_core.config import load
from lomas_core.contracts import ROBOT_SAY, STRANGER_SEEN, StrangerSeen
from lomas_core.errors import LomasError

from app import container, seed
from app.ears import Ears, matches
from app.greeter import Greeter, clean_name
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
    "content.author.cache_dir=data/test-lessons",
]


def build(*extra: str):
    cfg = load("config", "debug", [*HEADLESS, *extra], use_env=False)
    system = container.build(cfg, clock=FakeClock(), bus=container.event_bus(cfg))
    seed.real_class(system)
    return system


@pytest.fixture
def system():
    built = build("enrolment.by_voice=true")
    yield built
    built.close()


def hearing(system, said: str, clock=None):
    from tests.test_listener import FakeEars, FakeMic

    listener = Listener(system.cfg, system.bus, clock or system.clock, FakeMic(), FakeEars(said),
                        speakers=system.speakers)
    system.listener = listener
    return listener


def said(system) -> list[str]:
    return [u.text for _n, u in system.bus.replay(ROBOT_SAY)]


def phrasings(prompt: str) -> list[str]:
    """Every way the robot might say one thing. A test that asserts on one
    word instead fails whenever the dice pick another line."""
    import yaml

    from pathlib import Path

    body = Path(f"config/prompts/en/{prompt}.yaml").read_text(encoding="utf-8")
    return yaml.safe_load(body)["lines"]


# --- being told to begin, out loud ---------------------------------------


def test_what_counts_as_asking_for_a_class() -> None:
    """"Start the class" comes back as "start the clause" often enough that
    an exact match would make the robot look deaf."""
    phrases = ["start the class", "hey lomas"]

    assert matches("Start the class", phrases, 0.75) == "start the class"
    assert matches("start the clause", phrases, 0.75)
    assert matches("okay everyone, start the class please", phrases, 0.75)
    assert matches("what is chlorophyll", phrases, 0.75) == ""


def test_a_voice_in_the_room_starts_a_class(system) -> None:
    listener = hearing(system, "start the class", clock=RealClock())
    ears = Ears(system.cfg, system.bus, RealClock(), listener, runner=system.runner)

    ears.wait_for_a_class()
    deadline = time.monotonic() + 5
    while not system.runner.teaching and time.monotonic() < deadline:
        time.sleep(0.02)

    try:
        assert system.runner.teaching, "nobody could start this robot without a screen"
    finally:
        ears.stop()
        system.runner.stop()


def test_a_topic_asked_for_in_the_same_breath(system) -> None:
    """"Start the class on the solar system" is one sentence, and the class
    should not then be asked what to learn."""
    listener = hearing(system, "start the class on the solar system", clock=RealClock())
    ears = Ears(system.cfg, system.bus, RealClock(), listener, runner=system.runner)
    started = []
    system.runner.start = lambda topic="", language="", by="": started.append(topic)

    ears._begin("start the class on the solar system")

    assert started and "solar system" in started[0]


def test_the_phrase_is_config() -> None:
    cfg = load("config", "debug", [], use_env=False)
    assert "start the class" in cfg.flow.start_phrases

    quiet = load("config", "debug", ["flow.start_phrases=[]"], use_env=False)
    assert quiet.flow.start_phrases == [], "a school can have a robot that waits to be pressed"


def test_nothing_starts_a_second_class(system) -> None:
    system.runner.start()
    try:
        with pytest.raises(LomasError, match="already running"):
            system.runner.start()
    finally:
        system.runner.stop()


# --- meeting somebody new ------------------------------------------------


def test_a_name_out_of_a_sentence() -> None:
    lead_ins = ["my name is", "i am"]

    assert clean_name("My name is Akshay", lead_ins) == "Akshay"
    assert clean_name("i am meera patil", lead_ins) == "Meera Patil"
    assert clean_name("Akshay.", lead_ins) == "Akshay"
    assert clean_name("", lead_ins) == ""


def a_greeter(system, name_heard: str):
    listener = hearing(system, name_heard)
    enrolled = []

    class Sweep:
        def next_roll(self, scope):
            return "07"

        def start(self, scope, name, roll_no, granted_by, document_ref=""):
            enrolled.append(("start", name, roll_no, granted_by))
            return {"enrolment_id": "e1", "student_id": "s1", "name": name}

        def add_frame(self, enrolment_id):
            enrolled.append(("frame", enrolment_id))

        def finish(self, scope, enrolment_id):
            enrolled.append(("finish", enrolment_id))
            return {"vectors": 9}

        def cancel(self, enrolment_id):
            enrolled.append(("cancel", enrolment_id))

    greeter = Greeter(system.cfg, system.bus, system.clock, Sweep(), listener,
                      system.prompts,
                      say=lambda text: system.bus.publish(ROBOT_SAY, __import__(
                          "lomas_core.contracts", fromlist=["Utterance"]).Utterance(
                              text=text, language="en", reason="greeter")),
                      scope_of=lambda: system.orchestrator.scope)
    return greeter, enrolled


def test_the_robot_introduces_itself_and_enrols(system) -> None:
    greeter, enrolled = a_greeter(system, "My name is Akshay")

    greeter._meet(StrangerSeen(track_id=1, seen_for=5.0, source_id="head", at=1.0))

    steps = [step[0] for step in enrolled]
    assert steps[0] == "start" and steps[-1] == "finish"
    assert enrolled[0][1] == "Akshay", "enrolled as whatever sentence was said"
    assert "frame" in steps, "no pictures were taken"

    spoken = said(system)
    # Against the prompt file, not against a word: the phrasings are picked
    # at random and only two of the three ask_name lines say "name".
    assert spoken[0] in phrasings("ask_name"), "it never asked"
    assert "akshay" in " ".join(spoken).lower(), "it never said hello back"


def test_a_child_who_says_nothing_is_not_enrolled(system) -> None:
    greeter, enrolled = a_greeter(system, "")

    greeter._meet(StrangerSeen(track_id=1, seen_for=5.0, source_id="head", at=1.0))

    assert enrolled == [], "a silence is not consent and not a name"


def test_every_angle_is_coached(system) -> None:
    greeter, enrolled = a_greeter(system, "Akshay")

    greeter._meet(StrangerSeen(track_id=1, seen_for=5.0, source_id="head", at=1.0))

    spoken = " ".join(said(system)).lower()
    for angle in system.cfg.enrolment.required_angles:
        assert angle in spoken, f"nobody was asked to look {angle}"


def test_it_is_off_unless_a_school_turns_it_on() -> None:
    """Who may consent to a child's face being stored is a school's
    decision, so the default is a teacher pressing the button."""
    assert load("config", "debug", [], use_env=False).enrolment.by_voice is False
    assert load("config", "pi", [], use_env=False).enrolment.by_voice is True


def test_nobody_is_greeted_in_the_middle_of_a_class(system) -> None:
    greeter, enrolled = a_greeter(system, "Akshay")
    greeter.busy = lambda: True

    system.bus.publish(STRANGER_SEEN, StrangerSeen(track_id=1, seen_for=5.0,
                                                   source_id="head", at=1.0))
    time.sleep(0.1)

    assert enrolled == [], "a lesson was interrupted to ask somebody their name"
