"""Saying something without saying anything.

One microphone in a room of forty is the problem. A card held up is an
answer from a named child with nothing to transcribe; a hand held up is a
question asked without shouting over the lesson. The cards cost a couple of
milliseconds a frame and need no new dependency, which is why they are on by
default and the hand model is not.
"""
from __future__ import annotations

import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from lomas_core.clock import FakeClock
from lomas_core.config import load
from lomas_core.contracts import (
    CARD_SEEN,
    HAND_UP,
    QUIZ_ANSWERED,
    QUIZ_POSED,
    ROBOT_SAY,
    SIGN_SEEN,
    QuizPosed,
    Utterance,
)

from app import container, seed
from app.answering import Answering
from app.asking import Asking
from app.signs import SignWatch
from app.speaker.types import Heard
from app.speaker.resolvers.card import CardHeldUp
from lomas_signs import CARD_READERS
from lomas_signs.sheet import card as make_card
from lomas_signs.types import Box, Card, Sign

HEADLESS = [
    "storage.backend=memory",
    "vision.pipeline.enabled=false",
    "speech.tts.engine=null",
    "speech.stt.engine=keyboard",
    "speech.wake.engine=keyboard",
    "llm.provider=offline",
    "signs.enabled=true",
]

WHITE = 255


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


def picture(*cards, size: int = 240, turn: int = 0) -> np.ndarray:
    """A frame with these markers in it, laid out left to right."""
    faces = []
    for marker_id in cards:
        face = make_card("DICT_4X4_50", marker_id, f"#{marker_id}", size, ["A", "B", "C", "D"])
        for _ in range(turn):
            face = cv2.rotate(face, cv2.ROTATE_90_CLOCKWISE)
        faces.append(face)

    side = faces[0].shape[0] if faces else size
    gap = side // 10 or 1
    page = np.full((side + gap * 2, (side + gap) * max(len(faces), 1) + gap, 3), WHITE, np.uint8)
    for index, face in enumerate(faces):
        left = gap + index * (side + gap)
        page[gap:gap + side, left:left + side] = face
    return page


def frame(image: np.ndarray, seq: int = 1, ts: float = 1.0):
    return SimpleNamespace(image=image, seq=seq, ts=ts, source_id="head")


def seen(system, event: str) -> list:
    return [payload for _name, payload in system.bus.replay(event)]


# --- reading a card --------------------------------------------------------


def test_a_printed_card_is_read_back() -> None:
    cfg = load("config", "debug", [], use_env=False).signs.cards
    reader = CARD_READERS.create(cfg.reader, cfg)

    found = reader.read(picture(7))

    assert [c.marker_id for c in found] == [7]
    assert found[0].upright


def test_which_way_up_is_the_answer() -> None:
    """A square card has four ways up, and that is how forty children answer
    one question in one frame."""
    cfg = load("config", "debug", [], use_env=False).signs.cards
    reader = CARD_READERS.create(cfg.reader, cfg)

    for turn in range(4):
        found = reader.read(picture(7, turn=turn))
        assert found[0].turn == turn, f"a card turned {turn} read as {found[0].turn}"


def test_a_speck_across_the_room_is_not_a_card() -> None:
    cfg = load("config", "debug", ["signs.cards.min_size_px=400"], use_env=False).signs.cards
    reader = CARD_READERS.create(cfg.reader, cfg)

    assert reader.read(picture(7)) == []


def test_a_whole_class_reads_in_one_frame() -> None:
    cfg = load("config", "debug", [], use_env=False).signs.cards
    reader = CARD_READERS.create(cfg.reader, cfg)

    found = reader.read(picture(1, 2, 3, 4))

    assert sorted(c.marker_id for c in found) == [1, 2, 3, 4]


def test_the_cards_need_nothing_installed() -> None:
    """The reason cards come before hands: the marker dictionaries are built
    into OpenCV, so a card costs no dependency and no model file."""
    cfg = load("config", "debug", [], use_env=False).signs.cards

    assert CARD_READERS.create(cfg.reader, cfg).available


# --- a card belongs to a child ---------------------------------------------


def watching(system):
    return SignWatch(system.cfg, system.bus, system.clock, SimpleNamespace(start=lambda: None),
                     system.repos, scope_of=lambda: system.orchestrator.scope)


def with_a_card(system, marker_id: int = 3):
    student = system.repos["student"].list_for_class(system.orchestrator.scope)[0]
    system.repos["card"].issue(system.orchestrator.scope, student["id"], marker_id)
    return student


def test_the_card_says_who(system) -> None:
    student = with_a_card(system)
    watch = watching(system)

    for seq in range(3):
        watch.read(frame(picture(3), seq=seq + 1))

    cards = seen(system, CARD_SEEN)
    assert cards and cards[-1].student_id == student["id"]
    assert cards[-1].student_name == student["name"]
    assert cards[-1].answer == "A"


def test_a_card_nobody_owns_is_still_seen(system) -> None:
    """A spare card, or one handed out and not recorded. Worth reporting so
    a teacher can see why the robot is not naming anybody."""
    watch = watching(system)

    for seq in range(3):
        watch.read(frame(picture(42), seq=seq + 1))

    cards = seen(system, CARD_SEEN)
    assert cards and cards[-1].student_id == ""


def test_a_card_has_to_be_held_not_flashed(system) -> None:
    """One read is a card going into a bag. The same trick rules out a wave,
    which is motion by definition."""
    with_a_card(system)
    watch = watching(system)

    watch.read(frame(picture(3), seq=1))

    assert seen(system, CARD_SEEN) == []


def test_holding_a_card_up_is_asking(system) -> None:
    student = with_a_card(system)
    watch = watching(system)

    for seq in range(3):
        watch.read(frame(picture(3), seq=seq + 1))

    hands = seen(system, HAND_UP)
    assert hands and hands[-1].student_id == student["id"]
    assert hands[-1].by == "card"


def test_a_school_can_have_cards_that_only_answer() -> None:
    system = build("signs.cards.ask_when_upright=false")
    try:
        with_a_card(system)
        watch = watching(system)
        for seq in range(3):
            watch.read(frame(picture(3), seq=seq + 1))

        assert seen(system, CARD_SEEN)
        assert seen(system, HAND_UP) == []
    finally:
        system.close()


# --- a hand ----------------------------------------------------------------


class OneSign:
    """A hand reader that always sees the same sign, so the rules around it
    can be tested without a model."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.available = True
        self.reads = 0

    def read(self, image, at: float = 0.0):
        self.reads += 1
        return [Sign(name=self.name, box=Box(x=10, y=10, w=40, h=40), score=0.9, at=at)]

    def describe(self) -> str:
        return "a fake hand"

    def close(self) -> None:
        return None


def test_a_sign_means_what_the_school_says_it_means(system) -> None:
    watch = watching(system)
    watch.hands = OneSign("Pointing_Up")

    for seq in range(3):
        watch.read(frame(picture(), seq=seq + 1))

    signs = seen(system, SIGN_SEEN)
    assert signs and signs[-1].means == "ask"
    assert seen(system, HAND_UP)


def test_a_sign_nobody_mapped_is_ignored() -> None:
    system = build("signs.hands.actions={}")
    try:
        watch = watching(system)
        watch.hands = OneSign("Pointing_Up")

        for seq in range(3):
            watch.read(frame(picture(), seq=seq + 1))

        assert seen(system, SIGN_SEEN) == []
    finally:
        system.close()


def test_a_nod_can_answer_a_spoken_question(system) -> None:
    """The robot reads a topic back and asks "is that right?". A thumb is an
    answer, and a class that cannot make itself heard still has one."""
    watch = watching(system)
    watch.hands = OneSign("Thumb_Up")

    for seq in range(3):
        watch.read(frame(picture(), seq=seq + 1))

    assert watch.agreement(within_seconds=60.0) == "yes"


def test_hands_are_off_until_a_school_turns_them_on() -> None:
    """The expensive half. Cards cost milliseconds; a hand model costs tens,
    on a Pi that is already running a face detector."""
    cfg = load("config", "debug", [], use_env=False)

    assert cfg.signs.hands.reader == "none"
    assert load("config", "pi", [], use_env=False).signs.hands.reader == "none"
    assert load("config", "pi", [], use_env=False).signs.cards.reader == "aruco"


def test_a_missing_hand_model_is_not_a_broken_robot() -> None:
    from lomas_signs import HAND_READERS

    cfg = load("config", "debug", ["signs.hands.model=models/not-here.task"],
               use_env=False).signs.hands
    reader = HAND_READERS.create("mediapipe", cfg)

    assert reader.available is False
    assert reader.read(np.zeros((10, 10, 3), np.uint8)) == []


# --- whose voice it was ----------------------------------------------------


def test_a_card_held_up_outranks_a_guess() -> None:
    """The only signal in the chain a child gives on purpose."""
    cfg = load("config", "debug", [], use_env=False).speech.speaker
    resolver = CardHeldUp(cfg)

    found = resolver.resolve(Heard(text="it is six", cards=["s2"], visible=["s1", "s2"]), None)

    assert found is not None and found.student_id == "s2"


def test_two_cards_up_is_nobody_in_particular() -> None:
    cfg = load("config", "debug", [], use_env=False).speech.speaker
    resolver = CardHeldUp(cfg)

    assert resolver.resolve(Heard(text="x", cards=["s1", "s2"]), None) is None


# --- answering with a card -------------------------------------------------


def posed(system, options=("Sunlight", "Water", "Air", "Soil")):
    system.bus.publish(QUIZ_POSED, QuizPosed(session_id="s1", question_id="q1",
                                             text="what do leaves use?", options=tuple(options)))


def test_a_class_answers_in_one_frame(system) -> None:
    student = with_a_card(system)
    answering = Answering(system.cfg, system.bus, system.clock)
    watch = watching(system)
    posed(system)

    for seq in range(3):
        watch.read(frame(picture(3, turn=1), seq=seq + 1))

    answers = seen(system, QUIZ_ANSWERED)
    assert answers, "a card held up answered nothing"
    assert answers[-1].student_id == student["id"]
    assert answers[-1].response == "Water", "B should be the second option, not the letter"
    assert answering.recorded == 1


def test_a_card_held_steady_answers_once(system) -> None:
    with_a_card(system)
    Answering(system.cfg, system.bus, system.clock)
    watch = watching(system)
    posed(system)

    for seq in range(9):
        watch.read(frame(picture(3, turn=1), seq=seq + 1))

    assert len(seen(system, QUIZ_ANSWERED)) == 1


def test_a_card_up_when_nothing_was_asked_is_not_an_answer(system) -> None:
    with_a_card(system)
    Answering(system.cfg, system.bus, system.clock)
    watch = watching(system)

    for seq in range(3):
        watch.read(frame(picture(3, turn=1), seq=seq + 1))

    assert seen(system, QUIZ_ANSWERED) == []


# --- a child interrupting --------------------------------------------------


class Mic:
    def __init__(self) -> None:
        self.available = True
        self.turns: list[dict] = []

    def listen(self, **kwargs):
        self.turns.append(kwargs)
        return {"text": "why do leaves fall"}

    def describe(self) -> str:
        return "a fake mic"


class Mouth:
    """A voice that records being asked to stop and to carry on."""

    def __init__(self) -> None:
        self.yielded = 0
        self.resumed = 0

    def yield_now(self) -> None:
        self.yielded += 1

    def resume(self) -> None:
        self.resumed += 1


def asking(system, teaching=True):
    voice, mic = Mouth(), Mic()
    service = Asking(system.cfg, system.bus, system.clock, voice, mic, system.prompts,
                     teaching=lambda: teaching)
    return service, voice, mic


def hand_up(system, student_id: str = "s1", name: str = "Ananya") -> None:
    from lomas_core.contracts import HandUp

    system.bus.publish(HAND_UP, HandUp(student_id=student_id, student_name=name, by="card",
                                       at=system.clock.now()))
    time.sleep(0.2)


def test_the_robot_finishes_its_sentence_then_stops(system) -> None:
    """Not mid-word. A hole in the middle of a sentence is what makes a
    robot sound broken rather than polite."""
    service, voice, mic = asking(system)

    hand_up(system)

    assert voice.yielded == 1
    assert mic.turns, "nobody was listened to"
    assert voice.resumed == 1, "the lesson was never picked up again"


def test_the_child_is_asked_by_name(system) -> None:
    service, _voice, _mic = asking(system)

    hand_up(system, name="Ananya")

    said = " ".join(u.text for u in seen(system, ROBOT_SAY))
    assert "Ananya" in said or "listening" in said


def test_the_question_is_attributed_to_whoever_raised_it(system) -> None:
    """A card says who. The robot should not then guess, and certainly not
    ask "who was that?" of a child it just called by name."""
    _service, _voice, mic = asking(system)

    hand_up(system, student_id="s1")

    assert mic.turns[0]["student_id"] == "s1"
    assert mic.turns[0]["attribute"] is False


def test_one_child_at_a_time(system) -> None:
    _service, voice, mic = asking(system)

    for _ in range(4):
        hand_up(system, student_id="s1")

    assert len(mic.turns) <= system.cfg.signs.asking.per_step


def test_the_same_child_does_not_get_every_turn(system) -> None:
    """A cooldown, so the confident one does not have the lesson and the
    quiet ones never get asked."""
    service, _voice, mic = asking(system)

    hand_up(system, student_id="s1")
    service._this_step = 0          # pretend a new idea began
    hand_up(system, student_id="s1")

    assert len(mic.turns) == 1
    assert service.refused >= 1


def test_nothing_is_interrupted_when_no_class_is_running(system) -> None:
    _service, voice, mic = asking(system, teaching=False)

    hand_up(system)

    assert voice.yielded == 0 and mic.turns == []


def test_interrupting_can_be_switched_off() -> None:
    system = build("signs.asking.enabled=false")
    try:
        _service, voice, mic = asking(system)

        hand_up(system)

        assert voice.yielded == 0 and mic.turns == []
    finally:
        system.close()


# --- what it costs ---------------------------------------------------------


def test_reading_cards_is_cheap() -> None:
    """The number that decided the order of this whole feature: cards are
    milliseconds, a hand model is tens of them."""
    cfg = load("config", "debug", [], use_env=False).signs.cards
    reader = CARD_READERS.create(cfg.reader, cfg)
    image = picture(1, 2, 3)

    began = time.perf_counter()
    for _ in range(10):
        reader.read(image)
    each = (time.perf_counter() - began) / 10

    assert each < 0.1, f"a card read cost {each * 1000:.0f} ms"
