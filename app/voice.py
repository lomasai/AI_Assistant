from __future__ import annotations

import queue
import threading

from lomas_core import logging as log
from lomas_core.contracts import (
    ROBOT_SAY,
    ROBOT_SPOKE,
    ROBOT_YIELDED,
    SESSION_PAUSED,
    Utterance,
)
from lomas_core.errors import LomasError
from lomas_core.events import EventBus
from lomas_speech import DuplexGate, TextToSpeech
from lomas_speech.types import SpeechHandle

STOP = None
RESUMING = "resuming"
DEFAULT_LANGUAGE = "en"


def allow_everything(_text: str, _language: str, _session_id: str = "") -> bool:
    """The default filter. Named rather than implied, so nobody has to
    wonder later whether safety was missing or absent on purpose."""
    return True


class Voice:
    """Turns `robot.say` into sound, one sentence at a time.

    Everything goes through a single speaking thread, because a robot has one
    mouth. Found on the Pi: the tutor answering a question from a web request
    and the quiz asking the next question from the lesson spoke over each
    other, and the two shared one audio process until one of them found it
    gone.

    A lesson still waits for each sentence to finish before starting the next
    - that is `Utterance.blocking`, and it is the default. An agent answering
    from somewhere else does not: the teacher's Ask button used to hold the
    browser for sixteen seconds while the whole answer was read aloud.
    """

    def __init__(self, tts: TextToSpeech, gate: DuplexGate, bus: EventBus, guard=allow_everything,
                 wait_seconds: float = 0.0) -> None:
        self.tts = tts
        self.gate = gate
        self.bus = bus
        self.guard = guard
        self.wait_seconds = wait_seconds or None
        self.stop_seconds = self.wait_seconds
        self.language = DEFAULT_LANGUAGE
        self.log = log.get("voice")

        self._mute_reported = False
        self._stopping = False
        self._speaking = False
        # What the robot had not said when a child put a hand up.
        self.held = ""
        self._current: SpeechHandle | None = None
        self._queue: queue.Queue = queue.Queue()
        self._worker = threading.Thread(target=self._speak_loop, name="voice", daemon=True)
        self._worker.start()

        bus.subscribe(ROBOT_SAY, self._on_say)

        # Mid-sentence, not at the end of it. A teacher who has to wait out a
        # paragraph before the room goes quiet stops using the pause button.
        bus.subscribe(SESSION_PAUSED, self._on_pause)

    @property
    def busy(self) -> bool:
        """Speaking, or about to. A microphone opened while a sentence is
        still queued records the robot saying it: on the Pi a child's question
        came back as the lesson segment the robot was about to read."""
        return self._speaking or not self._queue.empty()

    def _on_say(self, _event: str, utterance: Utterance) -> None:
        # Asked before a sound is made. A filter that subscribes to an event
        # has already lost the race with the speaker.
        if self._stopping or not self.guard(utterance.text, utterance.language, utterance.session_id):
            return

        done = threading.Event()
        self._queue.put((utterance, done))
        if utterance.blocking:
            if not done.wait(self.wait_seconds):
                self.log.error("gave up waiting on a sentence: %s", utterance.text[:60])

    def _speak_loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is STOP:
                return
            utterance, done = item
            try:
                self._speak(utterance)
            finally:
                done.set()

    def _speak(self, utterance: Utterance) -> None:
        self._speaking = True
        self.gate.on_speech_start()
        handle: SpeechHandle | None = None
        try:
            handle = self.tts.speak(utterance.text, utterance.language)
            self._current = handle
            handle.wait()
            if handle.error:
                raise LomasError(handle.error)
        except LomasError as exc:
            # A player killed on the way out is shutting down, not broken.
            # No voice is not no lesson either: a missing piper binary makes
            # the robot quiet and says why, once - a warning on every
            # sentence is a log nobody reads.
            if not self._stopping and not self._mute_reported:
                self._mute_reported = True
                self.log.error("no voice, teaching silently: %s", exc)
        finally:
            self._speaking = False
            self._current = None
            self.gate.on_speech_end()
            self._keep_the_rest(handle, utterance)

        who = f" [{utterance.student_name}]" if utterance.student_name else ""
        self.log.info("%s%s", utterance.text, who)
        self.bus.publish(ROBOT_SPOKE, utterance)

    # --- letting a child in -----------------------------------------------

    def yield_now(self) -> None:
        """Finish this sentence, then stop and wait.

        Not mid-word: cutting off in the middle of a sentence leaves a hole
        in the lesson that everybody hears, and the robot sounds broken
        rather than polite.

        The queue is deliberately not blocked. What comes next is the robot
        turning to the child and the answer to their question, and a lesson
        that muzzled those would have stopped for nothing. Holding the
        *lesson* is the teaching step's job, and it does it already.
        """
        handle = self._current
        if handle is not None and not handle.done:
            handle.ask_to_yield()

    def resume(self) -> None:
        """Pick the lesson up where it stopped.

        Queued behind whatever was said to the child, because the voice is
        one mouth and its order is the order the room hears.
        """
        unsaid, self.held = self.held, ""
        if unsaid:
            self.bus.publish(ROBOT_SAY, Utterance(text=unsaid, language=self.language,
                                                  reason=RESUMING))

    def _keep_the_rest(self, handle: SpeechHandle | None, utterance: Utterance) -> None:
        if handle is None or not handle.left_to_say:
            return
        self.held = handle.left_to_say
        self.language = utterance.language or self.language
        self.bus.publish(ROBOT_YIELDED, {"session_id": utterance.session_id,
                                         "left": len(self.held)})
        self.log.info("stopped for a raised hand, %d characters left to say", len(self.held))

    def _on_pause(self, _event: str, _payload) -> None:
        # Queued sentences go too. Otherwise the next one starts the moment
        # the current one is cut off, and the pause did nothing audible.
        self._drain()
        self.tts.stop()
        # A paused class is not a class waiting on a child, and the held
        # remainder would otherwise arrive when it resumed.
        self.held = ""

    def _drain(self) -> None:
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                return
            if item is not STOP:
                item[1].set()

    def stop(self) -> None:
        self._stopping = True
        self._drain()
        self.tts.stop()
        self._queue.put(STOP)
        self._worker.join(timeout=self.stop_seconds)
        self.gate.on_speech_end()
