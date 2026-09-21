"""Any topic, and a class of real children.

A robot that teaches the one lesson somebody wrote a pack for is a demo, and
a class of five invented students is a demo too. These are the two things
that make a run real: whoever was enrolled at the robot, learning whatever
was asked for.
"""
from __future__ import annotations

import json

import pytest

from lomas_core.clock import FakeClock
from lomas_core.config import load
from lomas_core.errors import LomasError
from lomas_llm import Completion

from app import container, seed
from app.author import LessonWriter, as_json, slug

HEADLESS = [
    "storage.backend=memory",
    "vision.pipeline.enabled=false",
    "hardware.enabled=false",
    "speech.tts.engine=null",
    "speech.stt.engine=keyboard",
    "speech.wake.engine=keyboard",
    "llm.provider=offline",
    "flow.answer_wait_seconds=1",
    "flow.attendance_wait_seconds=1",
    "flow.tick_seconds=0.1",
]

WRITTEN = {
    "title": "The Solar System",
    "segments": [
        {"say": f"Part {n} of the solar system, said out loud.", "display": f"part {n}"}
        for n in range(1, 7)
    ],
    "questions": [
        {"ask": f"Question {n}?", "options": ["one", "two", "three"], "answer": 1}
        for n in range(1, 7)
    ],
}


def build(*extra: str):
    cfg = load("config", "debug", [*HEADLESS, *extra], use_env=False)
    return container.build(cfg, clock=FakeClock(), bus=container.event_bus(cfg))


def answering(system, body: dict, fenced: bool = False):
    text = json.dumps(body)
    if fenced:
        text = f"Here you go:\n```json\n{text}\n```"
    system.orchestrator.author.llm = type(
        "Stub", (), {"complete": lambda _s, *_a, **_k: Completion(text=text, provider="stub")}
    )()
    return system.orchestrator.author


@pytest.fixture
def system(tmp_path):
    built = build(f"content.author.cache_dir={tmp_path.as_posix()}")
    seed.real_class(built)
    yield built
    built.close()


# --- what comes back from a model -----------------------------------------


def test_a_fenced_reply_is_still_json() -> None:
    """Models fence their JSON and say hello first. Neither is a failure."""
    assert as_json('Here you go:\n```json\n{"title": "x"}\n```') == {"title": "x"}
    assert as_json('{"title": "x"}') == {"title": "x"}


def test_a_reply_with_no_json_says_so() -> None:
    with pytest.raises(LomasError, match="did not return JSON"):
        as_json("I am afraid I cannot do that")


def test_a_reply_that_ran_out_of_tokens_is_mended() -> None:
    """What the Pi got: a lesson cut off mid-sentence at the token limit. Five
    whole segments in front of a class beat an exception."""
    cut = '{"title": "Machine learning", "segments": [{"say": "One."}, {"say": "Tw'

    assert as_json(cut) == {"title": "Machine learning", "segments": [{"say": "One."}]}
    assert as_json('{"title": "x",}') == {"title": "x"}


def test_a_reply_with_nothing_usable_says_so() -> None:
    with pytest.raises(LomasError, match="could not be read"):
        as_json('{"title": ')


def test_a_topic_becomes_a_lesson_id() -> None:
    assert slug("The Solar System!") == "the-solar-system"


# --- writing a lesson ------------------------------------------------------


def test_any_topic_gets_a_lesson(system) -> None:
    author = answering(system, WRITTEN)

    lesson, quiz = author.write("solar system", "en")

    assert lesson.title == "The Solar System"
    assert len(lesson) == 6
    assert lesson.written is True, "a report must be able to tell this from a reviewed pack"
    assert [q.ask for q in quiz.questions][0] == "Question 1?"


def test_the_class_teaches_what_was_asked_for(system) -> None:
    answering(system, WRITTEN)

    ctx = system.orchestrator.open_session(topic="solar system")

    assert ctx.lesson.id == "solar-system"
    assert ctx.content.quiz_for("solar-system") is not None, "the quiz step looks it up here"


def test_a_reviewed_pack_is_used_before_anything_is_written(system) -> None:
    author = answering(system, WRITTEN)
    asked = []
    author.llm.complete = lambda *a, **k: asked.append(a) or Completion(text="{}", provider="stub")

    ctx = system.orchestrator.open_session(topic="photosynthesis")

    assert not asked, "the pack was already there"
    assert ctx.lesson.written is False


def test_the_same_topic_twice_is_written_once(system, tmp_path) -> None:
    """Kept on disk: the same topic tomorrow costs nothing and works with the
    internet down."""
    author = answering(system, WRITTEN)
    calls = []
    inner = author.llm.complete
    author.llm.complete = lambda *a, **k: calls.append(1) or inner(*a, **k)

    author.write("solar system", "en")
    again = author.cached("solar system", "en")

    assert len(calls) == 1
    assert again is not None and again[0].title == "The Solar System"
    assert (tmp_path / "en" / "solar-system.json").exists()


def test_a_lesson_with_no_parts_is_refused(system) -> None:
    author = answering(system, {"title": "Empty", "segments": []})

    with pytest.raises(LomasError, match="no parts"):
        author.write("nothing at all", "en")


def test_the_writer_can_be_turned_off(tmp_path) -> None:
    """A school that teaches only reviewed content is a config line."""
    system = build("content.author.enabled=false")
    try:
        seed.real_class(system)
        assert system.orchestrator.author is None
        with pytest.raises(LomasError, match="photosynthesis"):
            system.orchestrator.open_session(topic="solar system")
    finally:
        system.close()


def test_how_long_a_lesson_is_comes_from_config(system) -> None:
    system.cfg.content.author.segments = 3
    system.cfg.content.author.questions = 2
    author = answering(system, WRITTEN)

    lesson, quiz = author.write("solar system", "en")

    assert len(lesson) == 3
    assert len(quiz.questions) == 2


# --- a class of real children ---------------------------------------------


def test_a_real_class_has_no_invented_children(system) -> None:
    """--seed writes five names that would otherwise turn up in a pilot
    school's report. Without it there is still a class to enrol into."""
    scope = system.orchestrator.scope

    assert system.repos["class"].get(scope, scope.class_id) is not None
    assert system.repos["student"].list_for_class(scope) == []


def test_a_class_runs_with_nobody_on_the_roster(system) -> None:
    answering(system, WRITTEN)

    ctx = system.orchestrator.open_session(topic="solar system")

    assert ctx.roster == []
    assert len(ctx.lesson) == 6, "the lesson is ready for whoever walks in"


def test_an_enrolled_child_is_the_roster(system) -> None:
    scope = system.orchestrator.scope
    student_id = system.repos["student"].create(scope, "Akshay", "01")

    roster = system.orchestrator.open_session(topic="photosynthesis").roster

    assert [row["id"] for row in roster] == [student_id]
    assert [row["name"] for row in roster] == ["Akshay"]


def test_seeding_is_still_there_when_it_is_asked_for(system) -> None:
    seed.demo_class(system)

    names = [row["name"] for row in system.repos["student"].list_for_class(
        system.orchestrator.scope)]
    assert "Ananya Sharma" in names


# --- what a child actually says -------------------------------------------


def topic_of(said: str, system) -> str:
    from app.author import clean_topic

    return clean_topic(said, system.cfg.content.author)


def test_a_sentence_becomes_a_topic(system) -> None:
    """Straight off the Pi: the whole sentence went to the writer as the
    topic, and what came back was unreadable."""
    said = ("My name is Akshay So today we want to learn about machine learning "
            "So, let's go ahead and see")

    assert topic_of(said, system) == "machine learning"


def test_a_topic_that_is_already_a_topic_is_left_alone(system) -> None:
    assert topic_of("solar system", system) == "solar system"
    assert topic_of("the water cycle", system) == "the water cycle"


def test_the_phrasings_are_config(system) -> None:
    system.cfg.content.author.topic_lead_ins = ["padhna hai"]
    assert topic_of("mujhe padhna hai gravity", system) == "gravity"


def test_a_lesson_that_cannot_be_written_does_not_end_the_class(system, caplog) -> None:
    """A model returning nonsense in front of a class is a lesson on
    something else, not a traceback and an empty room."""
    author = answering(system, WRITTEN)
    author.llm.complete = lambda *a, **k: Completion(text="I cannot do that", provider="stub")

    with caplog.at_level("ERROR"):
        ctx = system.orchestrator.open_session(topic="machine learning")

    assert ctx.lesson.id == system.cfg.content.default_topic
    assert any("could not write a lesson" in r.getMessage() for r in caplog.records)


def test_a_child_who_introduces_themselves_is_not_the_topic(system) -> None:
    """The Pi wrote a lesson called "akshay machine learning"."""
    assert topic_of("My name is Akshay, today we want to learn about machine learning",
                    system) == "machine learning"
    assert topic_of("I am Akshay and I want to learn about the solar system",
                    system) == "the solar system"


def test_asking_for_a_new_lesson_is_not_part_of_the_topic(system) -> None:
    """The Pi wrote "can we start a new lesson on ai"."""
    assert topic_of("Can we start a new lesson on AI?", system) == "ai"
    assert topic_of("can we do a class on black holes", system) == "black holes"
