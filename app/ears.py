from __future__ import annotations

import threading

from lomas_core import logging as log
from lomas_core.clock import Clock
from lomas_core.contracts import (
    SESSION_CLOSED,
    STEP_ENTERED,
    STEP_EXITED,
    TOPIC_CHOSEN,
    TOPIC_REQUESTED,
    TopicChosen,
)
from lomas_core.errors import LomasError
from lomas_core.events import EventBus
from lomas_core.schema import Config

from app.author import clean_topic, is_a_topic

MICROPHONE = "microphone"
A_VOICE = "a voice in the room"
CONFIRM = "confirm_topic"
AGAIN = "ask_topic_again"
TOPIC = "topic"


class Ears:
    """The robot listening without being asked to.

    Press to talk is right for a quiz answer in a room of forty, where the
    teacher decides whose answer it is. It is wrong for "what shall we learn
    today?", where the robot has just asked a question and should be the one
    waiting. It is also wrong for a child two minutes into a conversation.

    So: the mic opens by itself while a step in `hands_free_steps` is
    running, and again whenever something asks for a topic. Everything about
    who spoke, and what is a question rather than noise, is the listener's
    job exactly as it is when a button starts it.
    """

    def __init__(self, cfg: Config, bus: EventBus, clock: Clock, listener, voice=None,
                 runner=None, prompts=None, say=None) -> None:
        self.cfg = cfg
        self.bus = bus
        self.clock = clock
        self.listener = listener
        self.voice = voice
        # Set by the container. Without it the robot still teaches; it just
        # has to be told to start by the teacher's screen.
        self.runner = runner
        # For reading a topic back before a whole class is written about it.
        # Without either of these the robot takes the first thing it hears,
        # which is what it did before.
        self.prompts = prompts
        self._say = say
        self.log = log.get("ears")

        self.turns = 0
        self._step = ""
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        bus.subscribe(TOPIC_REQUESTED, self._on_topic_wanted)
        bus.subscribe(STEP_ENTERED, self._on_step)
        bus.subscribe(STEP_EXITED, self._on_step_done)
        bus.subscribe(SESSION_CLOSED, lambda *_: self.stop())

    @property
    def listening(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def stop(self) -> None:
        self._stop.set()
        self._step = ""

    # --- waiting to be told to begin --------------------------------------

    def wait_for_a_class(self) -> None:
        """Listen for somebody asking for a class, until one starts.

        This is what makes the robot a robot rather than a program with a
        web page: it is switched on, it waits, and a child or a teacher says
        "start the class" out loud.
        """
        if not self._usable() or self.runner is None or not self.cfg.flow.start_phrases:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._wait_loop, name="ears-idle", daemon=True)
        self._thread.start()
        self.log.info("say \"%s\" to begin", self.cfg.flow.start_phrases[0])

    def _wait_loop(self) -> None:
        while not self._stop.is_set():
            if self.runner.teaching:
                # A class is running and something else is listening for the
                # children; two microphones open at once hear each other.
                self.clock.sleep(self.cfg.flow.start_poll_seconds)
                continue

            heard = self._listen("", as_question=False, attribute=False)
            if self._stop.is_set():
                break
            asked = matches(heard, self.cfg.flow.start_phrases, self.cfg.speech.speaker.name_match)
            if asked:
                self.log.info("heard %r; starting a class", heard)
                self._begin(heard)

    def _begin(self, heard: str) -> None:
        topic = clean_topic(heard, self.cfg.content.author)
        # Only what is left after the asking. "Start the class" on its own
        # leaves nothing, which is the robot asking the class what to learn.
        wanted = "" if matches(topic, self.cfg.flow.start_phrases,
                               self.cfg.speech.speaker.name_match) else topic
        try:
            self.runner.start(wanted, by=A_VOICE)
        except LomasError as exc:
            self.log.info("not starting: %s", exc)

    # --- a topic, asked for out loud --------------------------------------

    def _on_topic_wanted(self, _event: str, wanted: TopicChosen) -> None:
        if not self._usable():
            return
        threading.Thread(target=self._hear_topic, args=(wanted.session_id,),
                         name="ears-topic", daemon=True).start()

    def _hear_topic(self, session_id: str) -> None:
        """Ask, and keep asking until there is a subject worth writing about.

        On the Pi this heard "can we start topic on...", found nothing in it
        that was a subject, and wrote a lesson on the water cycle anyway,
        because a model handed nothing invents something. Now what is left
        after the asking has to look like a topic, and the room hears it read
        back before a class is built on it.
        """
        for attempt in range(self.cfg.flow.topic_tries):
            topic = self._a_topic(session_id, attempt)
            if not topic or not self._agreed(session_id, topic):
                continue

            self.log.info("the class asked for: %s", topic)
            self.bus.publish(TOPIC_CHOSEN,
                             TopicChosen(session_id=session_id, text=topic, by=MICROPHONE))
            return

        # Said out loud, so the step is not left waiting out its whole timer
        # in silence for an answer that is not coming.
        self.log.info("nobody said what to teach")
        self.bus.publish(TOPIC_CHOSEN,
                         TopicChosen(session_id=session_id, text="", by=MICROPHONE))

    def _a_topic(self, session_id: str, attempt: int) -> str:
        if attempt:
            self._line(AGAIN)
        heard = self._listen(session_id, as_question=False, attribute=False, patience=TOPIC)
        topic = clean_topic(heard, self.cfg.content.author) if heard else ""
        if not topic or not is_a_topic(topic, self.cfg.content.author):
            self.log.info("no subject in %r", heard)
            return ""
        return topic

    def _agreed(self, session_id: str, topic: str) -> bool:
        """The topic, read back. A silence is a yes - a robot that demands an
        answer before it will teach is worse than one that mishears."""
        if not self.cfg.flow.confirm_topic or self.prompts is None or self._say is None:
            return True

        self._line(CONFIRM, topic=topic)
        heard = self._listen(session_id, as_question=False, attribute=False, patience="confirm")
        if matches(heard, self.cfg.flow.no_phrases, self.cfg.speech.speaker.name_match):
            self.log.info("not %r then", topic)
            return False
        return True

    def _line(self, prompt: str, **values) -> None:
        if self.prompts is None or self._say is None:
            return
        try:
            self._say(self.prompts.line(prompt, self.cfg.content.language, **values))
        except Exception as exc:  # a missing prompt file must not end the asking
            self.log.debug("no %s prompt: %s", prompt, exc)

    # --- a conversation, with nobody pressing anything --------------------

    def _on_step(self, _event: str, step) -> None:
        self._step = step.step
        # Whatever was listening for "start the class" stops now: two open
        # microphones in one room hear each other.
        self._stop.set()
        if not self._usable() or step.step not in self.cfg.speech.audio.hands_free_steps:
            return

        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, args=(step.session_id, step.step),
                                        name="ears", daemon=True)
        self._thread.start()

    def _on_step_done(self, _event: str, step) -> None:
        if step.step == self._step:
            self.stop()

    def _loop(self, session_id: str, step: str) -> None:
        """Listen, publish, listen again, until the step ends.

        Each turn is the same call the teacher's button makes, so a question
        asked this way is attributed, filtered for noise and answered exactly
        as a pressed one is.
        """
        self.log.info("listening for questions during %s", step)
        while not self._stop.is_set():
            self._listen(session_id, as_question=True, attribute=True)
            # Checked again before the gap: a microphone that has just been
            # unplugged should not leave a thread sleeping on it.
            if self._stop.is_set():
                break
            self.clock.sleep(self.cfg.speech.audio.hands_free_gap_seconds)

    # --- the one place a turn is taken ------------------------------------

    def _listen(self, session_id: str, as_question: bool, attribute: bool,
                patience: str = "") -> str:
        self.turns += 1
        try:
            heard = self.listener.listen(session_id=session_id, as_question=as_question,
                                         attribute=attribute, patience=patience)
        except LomasError as exc:
            # A microphone that has been unplugged mid-class is not a class
            # that stops; it is a class the teacher drives by hand again.
            self.log.error("stopped listening: %s", exc)
            self._stop.set()
            return ""
        return heard.get("text", "")

    def _usable(self) -> bool:
        return bool(self.cfg.speech.audio.hands_free and self.listener is not None
                    and self.listener.available)


def matches(heard: str, phrases: list[str], closeness: float) -> str:
    """Whether what was heard is one of these phrases, loosely.

    "Start the class" comes back as "start the clause" often enough that an
    exact match would make the robot look deaf.
    """
    from difflib import SequenceMatcher

    said = " ".join(heard.lower().split())
    if not said:
        return ""
    for phrase in phrases:
        wanted = phrase.lower()
        if wanted in said or SequenceMatcher(None, said, wanted).ratio() >= closeness:
            return phrase
    return ""
