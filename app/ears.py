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

from app.author import clean_topic

MICROPHONE = "microphone"


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

    def __init__(self, cfg: Config, bus: EventBus, clock: Clock, listener, voice=None) -> None:
        self.cfg = cfg
        self.bus = bus
        self.clock = clock
        self.listener = listener
        self.voice = voice
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

    # --- a topic, asked for out loud --------------------------------------

    def _on_topic_wanted(self, _event: str, wanted: TopicChosen) -> None:
        if not self._usable():
            return
        threading.Thread(target=self._hear_topic, args=(wanted.session_id,),
                         name="ears-topic", daemon=True).start()

    def _hear_topic(self, session_id: str) -> None:
        heard = self._listen(session_id, as_question=False, attribute=False)
        topic = clean_topic(heard, self.cfg.content.author) if heard else ""
        if not topic:
            self.log.info("nobody said what to teach")
            return

        self.log.info("the class asked for: %s", topic)
        self.bus.publish(TOPIC_CHOSEN,
                         TopicChosen(session_id=session_id, text=topic, by=MICROPHONE))

    # --- a conversation, with nobody pressing anything --------------------

    def _on_step(self, _event: str, step) -> None:
        self._step = step.step
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

    def _listen(self, session_id: str, as_question: bool, attribute: bool) -> str:
        self.turns += 1
        try:
            heard = self.listener.listen(session_id=session_id, as_question=as_question,
                                         attribute=attribute)
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
