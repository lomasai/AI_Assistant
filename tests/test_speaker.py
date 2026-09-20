"""Who just spoke.

Tapping a name works and is always right, but a teacher cannot tap for every
sentence of a conversation. These are the ways the robot works it out for
itself, and the order they are tried in - which is config, so most of what
follows is about that list behaving the way the file says it does.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from lomas_core.clock import FakeClock
from lomas_core.config import load
from lomas_core.contracts import (
    QUESTION_ASKED,
    ROBOT_SAY,
    STUDENT_IDENTIFIED,
    VISION_TRACKS,
    StudentIdentified,
    TracksSeen,
    TrackView,
)

from app import container, seed
from app.listener import Listener
from app.speaker import RESOLVERS, Room, SpeakerChain
from app.web.server import create_app

HEADLESS = [
    "storage.backend=memory",
    "vision.pipeline.enabled=false",
    "hardware.enabled=false",
    "speech.tts.engine=null",
    "speech.stt.engine=keyboard",
    "speech.wake.engine=keyboard",
    "llm.provider=offline",
    "flow.answer_wait_seconds=1",
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


@pytest.fixture
def roster(system):
    system.orchestrator.open_session()
    return system.repos["student"].list_for_class(system.orchestrator.scope)


def sees(system, student_id: str) -> None:
    """The camera reporting a recognised face, as the pipeline would."""
    system.bus.publish(
        VISION_TRACKS,
        TracksSeen(
            source_id="head", zone="front", at=system.clock.now(), width=1280, height=720,
            tracks=(TrackView(track_id=1, x=10, y=10, w=200, h=200, student_id=student_id,
                              attention=0.9, yaw=0.0, pitch=0.0, seen_for=3.0),),
        ),
    )


# --- the chain ------------------------------------------------------------


def test_the_order_is_the_config(system) -> None:
    assert system.speakers.names() == ["tapped", "single_face", "recent", "spoken_name", "ask"]


def test_a_resolver_can_be_switched_off_in_config() -> None:
    """The point of the list. One way at a time is how mouth movement gets
    tested without the rest of the chain answering for it."""
    system = build("speech.speaker.resolvers=[tapped]")
    try:
        assert system.speakers.names() == ["tapped"]
        assert system.speakers.resolve("why do leaves fall").how == "unknown"
    finally:
        system.close()


def test_every_resolver_named_in_the_default_config_exists() -> None:
    cfg = load("config", "debug", [], use_env=False)
    for name in cfg.speech.speaker.resolvers:
        assert name in RESOLVERS.keys()


def test_the_teacher_tap_beats_everything(system, roster) -> None:
    sees(system, roster[1]["id"])

    found = system.speakers.resolve("why do leaves fall", tapped=(roster[0]["id"], ""))

    assert found.student_id == roster[0]["id"]
    assert found.how == "tapped"
    assert found.name == roster[0]["name"], "the name is filled in from the roster"


def test_one_known_face_needs_no_name_at_all(system, roster) -> None:
    """The demo: one child in front of the camera, nobody taps anything."""
    sees(system, roster[2]["id"])

    found = system.speakers.resolve("why do leaves fall")

    assert found.student_id == roster[2]["id"]
    assert found.how == "single_face"


def test_two_faces_are_not_guessed_at(system, roster) -> None:
    sees(system, roster[0]["id"])
    sees(system, roster[1]["id"])

    found = system.speakers.resolve("why do leaves fall")

    assert found.how != "single_face"


def test_a_face_that_left_stops_counting(system, roster) -> None:
    sees(system, roster[0]["id"])
    system.clock.advance(system.cfg.speech.speaker.visible_seconds + 1)

    assert system.speakers.room.visible() == []


def test_identification_alone_counts_as_present(system, roster) -> None:
    """Recognition fires once per track; a track carrying no id afterwards
    must not read as an empty room."""
    system.bus.publish(STUDENT_IDENTIFIED, StudentIdentified(
        student_id=roster[0]["id"], track_id=4, source_id="head", zone="front", at=0.0))

    assert system.speakers.room.visible() == [roster[0]["id"]]


# --- saying your name -----------------------------------------------------


def only_names(system):
    """The chain with the camera taken out of it, which is what a test of
    names should be looking at."""
    return SpeakerChain(system.cfg, system.bus, system.clock, system.prompts, system.repos,
                        room=None, scope_of=lambda: system.orchestrator.scope)


def test_a_name_at_the_front_is_who_is_speaking(system, roster) -> None:
    first = roster[0]["name"].split()[0]

    found = only_names(system).resolve(f"I am {first}, why do leaves fall in winter?")

    assert found.student_id == roster[0]["id"]
    assert found.how == "spoken_name"


def test_the_introduction_is_taken_off_the_question(system, roster) -> None:
    """The tutor should be asked the question, not told who is asking."""
    first = roster[0]["name"].split()[0]

    found = only_names(system).resolve(f"I am {first}, my question is why do leaves fall?")

    assert found.text == "why do leaves fall?"


def test_a_misheard_name_still_finds_the_child(system, roster) -> None:
    """Whisper returns Akshaya, Akash and action for one child. A robot that
    accepts only the register's spelling is one nobody can talk to."""
    first = roster[0]["name"].split()[0]
    misheard = first[:-1] + "a" if len(first) > 3 else first + "a"

    found = only_names(system).resolve(f"my name is {misheard} and what is chlorophyll")

    assert found.student_id == roster[0]["id"]


def test_how_loose_the_matching_is_comes_from_config(system, roster) -> None:
    strict = build("speech.speaker.resolvers=[spoken_name]", "speech.speaker.name_match=1.0")
    try:
        first = roster[0]["name"].split()[0]
        strict.orchestrator.open_session()
        assert strict.speakers.resolve(f"I am {first}xyz, why do leaves fall").how == "unknown"
    finally:
        strict.close()


def test_a_name_inside_the_question_is_not_the_speaker(system, roster) -> None:
    """A lesson about Meera's garden is not Meera speaking."""
    first = roster[1]["name"].split()[0]

    found = only_names(system).resolve(f"why does the tree in {first} garden lose its leaves")

    assert found.how == "unknown"


# --- holding the turn -----------------------------------------------------


def test_the_name_is_said_once_and_then_it_is_a_conversation(system, roster) -> None:
    chain = only_names(system)
    first = roster[0]["name"].split()[0]
    chain.resolve(f"I am {first}, why do leaves fall?")

    again = chain.resolve("and does it grow them back?")

    assert again.student_id == roster[0]["id"]
    assert again.how == "recent"


def test_the_turn_runs_out(system, roster) -> None:
    chain = only_names(system)
    first = roster[0]["name"].split()[0]
    chain.resolve(f"I am {first}, why do leaves fall?")
    system.clock.advance(system.cfg.speech.speaker.recent_seconds + 1)

    assert chain.resolve("and does it grow them back?").how == "unknown"


def test_somebody_else_in_front_of_the_camera_takes_the_turn(system, roster) -> None:
    sees(system, roster[0]["id"])
    system.speakers.resolve("why do leaves fall")
    sees(system, roster[1]["id"])
    system.clock.advance(system.cfg.speech.speaker.visible_seconds + 1)
    sees(system, roster[1]["id"])

    found = system.speakers.resolve("and does it grow them back?")

    assert found.student_id == roster[1]["id"]


def test_a_new_class_starts_with_nobody_speaking(system, roster) -> None:
    chain = only_names(system)
    first = roster[0]["name"].split()[0]
    chain.resolve(f"I am {first}, why do leaves fall?")

    system.orchestrator.close_session()
    system.orchestrator.open_session()

    assert chain.resolve("and does it grow them back?").how == "unknown"


# --- asking, and only as a last resort ------------------------------------


def test_the_robot_asks_when_it_cannot_tell(system, roster) -> None:
    system.speakers.resolve("why do leaves fall in winter?")

    import yaml

    lines = yaml.safe_load(Path("config/prompts/en/who_is_asking.yaml").read_text())["lines"]
    asked = [u.text for _n, u in system.bus.replay(ROBOT_SAY) if u.reason == "ask"]
    assert len(asked) == 1
    assert asked[0] in lines, "the wording is a content edit, not code"


def test_nobody_is_interrupted_when_the_robot_already_knows(system, roster) -> None:
    sees(system, roster[0]["id"])

    system.speakers.resolve("why do leaves fall in winter?")

    assert not [u for _n, u in system.bus.replay(ROBOT_SAY) if u.reason == "ask"]


# --- through the microphone and the web -----------------------------------


def test_the_press_is_attributed_without_a_tap(system, roster) -> None:
    from tests.test_listener import FakeEars, FakeMic

    first = roster[0]["name"].split()[0]
    system.listener = Listener(system.cfg, system.bus, system.clock, FakeMic(),
                               FakeEars(f"I am {first}, why do leaves fall?"),
                               speakers=system.speakers)

    with TestClient(create_app(system)) as client:
        body = client.post("/api/listen", json={}).json()

    asked = [p for _n, p in system.bus.replay(QUESTION_ASKED)]
    assert body["student_id"] == roster[0]["id"]
    assert body["how"] == "spoken_name"
    assert asked[-1].student_name == roster[0]["name"]
    assert asked[-1].text == "why do leaves fall?", "the tutor is asked the question"


def test_nobody_is_asked_who_said_the_topic(system) -> None:
    """The robot answered "Who was that? Say your name" to a child saying
    what they wanted to learn. A topic has no speaker."""
    from tests.test_listener import FakeEars, FakeMic

    system.listener = Listener(system.cfg, system.bus, system.clock, FakeMic(),
                               FakeEars("today we want to learn about machine learning"),
                               speakers=system.speakers)

    with TestClient(create_app(system)) as client:
        body = client.post("/api/listen", json={"as_topic": True}).json()

    assert body["text"] == "machine learning", "the subject, not the sentence"
    assert not [u for _n, u in system.bus.replay(ROBOT_SAY) if u.reason == "ask"]
    assert not system.bus.replay(QUESTION_ASKED)
