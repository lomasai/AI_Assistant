"""What the class asked for, and what the robot then believes it is teaching.

Both halves of one bad class on the Pi. A child said "can we start topic
on..." and paused; the turn ended there, nothing in what was left was a
subject, and the writer - asked for a lesson on nothing - invented the water
cycle. The robot then taught the water cycle while the tutor and the quiz
went on believing the lesson was about leaves, because the session row still
said so and that row is what every agent reads.
"""
from __future__ import annotations

import pytest

from lomas_core.clock import FakeClock
from lomas_core.config import load
from lomas_core.contracts import LESSON_CHANGED, TOPIC_CHOSEN, TOPIC_REQUESTED, TopicChosen

from app import container, seed
from app.author import clean_topic, is_a_topic
from app.ears import Ears
from app.flow.steps.topic import Topic

HEADLESS = [
    "storage.backend=memory",
    "vision.pipeline.enabled=false",
    "speech.tts.engine=null",
    "speech.stt.engine=keyboard",
    "speech.wake.engine=keyboard",
    "llm.provider=offline",
    "flow.tick_seconds=0.05",
    "content.author.cache_dir=data/test-lessons",
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


class Written:
    """A writer that returns a lesson for whatever it is handed."""

    def __init__(self, system, title: str = "The water cycle") -> None:
        self.system = system
        self.title = title

    def cached(self, topic, language):
        return None

    def write(self, topic, language):
        from app.content import Lesson, Quiz, QuizQuestion, Segment

        lesson = Lesson(id="water-cycle", title=self.title, language=language, written=True,
                        segments=(Segment(id="s1", say="Water evaporates.", display="Water"),))
        quiz = Quiz(id="water-cycle-quiz", lesson="water-cycle", language=language,
                    questions=(QuizQuestion(id="q1", ask="Where does rain come from?"),))
        return lesson, quiz


# --- what counts as a topic ------------------------------------------------


def test_the_sentence_that_cost_a_class() -> None:
    """Heard on the Pi, word for word. Every lead-in comes off it and what
    is left is still not a subject."""
    cfg = load("config", "debug", [], use_env=False).content.author

    left = clean_topic("Can we start topic on...", cfg)

    assert not is_a_topic(left, cfg), "this went to the writer as a subject"


def test_a_real_topic_is_kept() -> None:
    cfg = load("config", "debug", [], use_env=False).content.author

    for said in ["can we learn about the water cycle",
                 "today we want to learn about machine learning",
                 "photosynthesis"]:
        topic = clean_topic(said, cfg)
        assert is_a_topic(topic, cfg), f"{said!r} became {topic!r} and was refused"


def test_what_a_filler_is_is_config() -> None:
    cfg = load("config", "debug", ["content.author.topic_fillers=[]"], use_env=False).content.author

    assert is_a_topic("can we start topic on", cfg), "a school can switch the check off"


# --- one source of truth ---------------------------------------------------


def test_the_session_record_follows_the_lesson(system) -> None:
    """The bug: a class taught the water cycle while every agent read the
    session row and answered about leaves."""
    ctx = system.orchestrator.open_session()
    ctx.notes["author"] = Written(system)
    step = Topic(system.cfg)

    step.enter(ctx)
    ctx.notes["topic_chosen"] = "water cycle"
    step.exit(ctx)

    row = system.repos["session"].get(ctx.scope, ctx.session_id)
    assert row["topic"] == "water-cycle", "the robot teaches one lesson and answers about another"
    assert ctx.lesson.id == "water-cycle"


def test_the_agents_are_told_what_is_being_taught(system) -> None:
    """The assembler is the only route an agent has to the lesson, and it
    reads the row - so this is the test that would have caught it."""
    from app.context.assembler import ContextAssembler

    ctx = system.orchestrator.open_session()
    ctx.notes["author"] = Written(system)
    step = Topic(system.cfg)
    step.enter(ctx)
    ctx.notes["topic_chosen"] = "water cycle"
    step.exit(ctx)

    assembler = ContextAssembler(system.cfg, system.repos, system.content)
    for_tutor = assembler.for_agent("tutor", ctx.scope, ctx.session_id)

    assert for_tutor.lesson_title == "The water cycle"


def test_a_surface_hears_that_the_lesson_changed(system) -> None:
    ctx = system.orchestrator.open_session()
    ctx.notes["author"] = Written(system)
    step = Topic(system.cfg)
    step.enter(ctx)
    ctx.notes["topic_chosen"] = "water cycle"
    step.exit(ctx)

    changed = [p for _n, p in system.bus.replay(LESSON_CHANGED)]

    assert changed and changed[-1].title == "The water cycle"
    assert changed[-1].written is True, "a written lesson must never look reviewed"


def test_nothing_changes_when_nobody_chose(system) -> None:
    ctx = system.orchestrator.open_session()
    was = ctx.lesson.id
    ctx.notes["author"] = Written(system)
    step = Topic(system.cfg)

    step.enter(ctx)
    step.exit(ctx)

    assert ctx.lesson.id == was
    assert system.repos["session"].get(ctx.scope, ctx.session_id)["topic"] == was


# --- the asking ------------------------------------------------------------


def a_room(system, *said: str):
    """A listener that hears these things in order, then silence."""
    heard = list(said)

    class Mic:
        available = True
        patience: list[str] = []

        def listen(self, **kwargs):
            Mic.patience.append(kwargs.get("patience", ""))
            return {"text": heard.pop(0) if heard else ""}

        def describe(self):
            return "fake"

    mic = Mic()
    Mic.patience = []
    spoken: list[str] = []
    ears = Ears(system.cfg, system.bus, system.clock, mic, prompts=system.prompts,
                say=spoken.append)
    return ears, spoken, Mic.patience


def chosen(system) -> list[str]:
    return [p.text for _n, p in system.bus.replay(TOPIC_CHOSEN)]


def test_a_topic_is_read_back_before_a_class_is_written(system) -> None:
    ears, spoken, _ = a_room(system, "can we learn about the water cycle", "yes")

    ears._hear_topic("s1")

    assert any("water cycle" in line for line in spoken), "it never read the topic back"
    assert chosen(system) == ["the water cycle"]


def test_a_no_makes_it_ask_again(system) -> None:
    ears, spoken, _ = a_room(system, "the water cycle", "no", "photosynthesis", "yes")

    ears._hear_topic("s1")

    assert chosen(system) == ["photosynthesis"], "it taught what the class said no to"


def test_silence_is_taken_as_agreement(system) -> None:
    """A robot that will not teach until somebody says the word yes is worse
    than one that mishears."""
    ears, _spoken, _ = a_room(system, "the water cycle", "")

    ears._hear_topic("s1")

    assert chosen(system) == ["the water cycle"]


def test_a_sentence_with_no_subject_is_asked_about_again(system) -> None:
    ears, spoken, _ = a_room(system, "Can we start topic on...", "the solar system", "yes")

    ears._hear_topic("s1")

    assert chosen(system) == ["the solar system"]
    assert len(spoken) >= 2, "it never asked a second time"


def test_giving_up_is_said_out_loud(system) -> None:
    """An empty choice, so the step is not left waiting out its timer in
    silence for an answer that is not coming."""
    ears, _spoken, _ = a_room(system, "", "")

    ears._hear_topic("s1")

    assert chosen(system) == [""]


def test_the_topic_turn_gets_more_patience(system) -> None:
    ears, _spoken, patience = a_room(system, "the water cycle", "yes")

    ears._hear_topic("s1")

    assert patience[0] == "topic", "the child who pauses mid-sentence is cut off again"
    assert system.cfg.speech.audio.patience_ms["topic"] > \
        system.cfg.speech.audio.stop_after_silence_ms


def test_reading_it_back_can_be_switched_off() -> None:
    system = build("flow.confirm_topic=false")
    try:
        ears, spoken, _ = a_room(system, "the water cycle")

        ears._hear_topic("s1")

        assert chosen(system) == ["the water cycle"]
        assert spoken == [], "it read the topic back anyway"
    finally:
        system.close()


def test_the_step_stops_waiting_once_the_room_has_given_up(system) -> None:
    ctx = system.orchestrator.open_session()
    step = Topic(system.cfg)
    step.enter(ctx)

    system.bus.publish(TOPIC_CHOSEN, TopicChosen(session_id=ctx.session_id, text="", by="mic"))

    from app.flow.states import StepResult
    assert step.tick(ctx, system.clock.now()) is StepResult.DONE
    step.exit(ctx)


def test_the_room_is_asked_only_after_the_question_has_been_heard(system) -> None:
    """The microphone opens when the class has heard what it is being asked,
    not while the robot is still asking it."""
    ctx = system.orchestrator.open_session()
    step = Topic(system.cfg)
    order: list[str] = []
    system.bus.subscribe("robot.say", lambda _n, _p: order.append("asked"))
    system.bus.subscribe(TOPIC_REQUESTED, lambda _n, _p: order.append("listening"))

    step.enter(ctx)
    step.exit(ctx)

    assert order == ["asked", "listening"]
