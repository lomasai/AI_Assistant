from __future__ import annotations

import queue
import threading

from lomas_core import logging as log
from lomas_core.contracts import ROBOT_SAY, ROBOT_SPOKE, SESSION_PAUSED, Utterance
from lomas_core.errors import LomasError
from lomas_core.events import EventBus
from lomas_speech import DuplexGate, TextToSpeech

STOP = None


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
        self.log = log.get("voice")

        self._mute_reported = False
        self._stopping = False
        self._queue: queue.Queue = queue.Queue()
        self._worker = threading.Thread(target=self._speak_loop, name="voice", daemon=True)
        self._worker.start()

        bus.subscribe(ROBOT_SAY, self._on_say)

        # Mid-sentence, not at the end of it. A teacher who has to wait out a
        # paragraph before the room goes quiet stops using the pause button.
        bus.subscribe(SESSION_PAUSED, self._on_pause)

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
        self.gate.on_speech_start()
        try:
            handle = self.tts.speak(utterance.text, utterance.language)
            handle.wait()
        except LomasError as exc:
            # A player killed on the way out is shutting down, not broken.
            # No voice is not no lesson either: a missing piper binary makes
            # the robot quiet and says why, once - a warning on every
            # sentence is a log nobody reads.
            if not self._stopping and not self._mute_reported:
                self._mute_reported = True
                self.log.error("no voice, teaching silently: %s", exc)
        finally:
            self.gate.on_speech_end()

        who = f" [{utterance.student_name}]" if utterance.student_name else ""
        self.log.info("%s%s", utterance.text, who)
        self.bus.publish(ROBOT_SPOKE, utterance)

    def _on_pause(self, _event: str, _payload) -> None:
        # Queued sentences go too. Otherwise the next one starts the moment
        # the current one is cut off, and the pause did nothing audible.
        self._drain()
        self.tts.stop()

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
