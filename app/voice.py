from __future__ import annotations

from lomas_core import logging as log
from lomas_core.errors import LomasError
from lomas_core.contracts import ROBOT_SAY, ROBOT_SPOKE, SESSION_PAUSED, Utterance
from lomas_core.events import EventBus
from lomas_speech import DuplexGate, TextToSpeech


def allow_everything(_text: str, _language: str, _session_id: str = "") -> bool:
    """The default filter. Named rather than implied, so nobody has to
    wonder later whether safety was missing or absent on purpose."""
    return True


class Voice:
    """Turns `robot.say` into sound, and mutes the microphones while it does.

    Steps publish; this is the only thing in the system holding a speaker.
    Publishing is synchronous, so a step that says something has finished
    saying it when `say` returns.
    """

    def __init__(self, tts: TextToSpeech, gate: DuplexGate, bus: EventBus, guard=allow_everything) -> None:
        self.tts = tts
        self.gate = gate
        self.bus = bus
        self.guard = guard
        self._mute_reported = False
        self.log = log.get("voice")
        bus.subscribe(ROBOT_SAY, self._on_say)

        # Mid-sentence, not at the end of it. A teacher who has to wait out a
        # paragraph before the room goes quiet stops using the pause button.
        bus.subscribe(SESSION_PAUSED, self._on_pause)

    def _on_say(self, _event: str, utterance: Utterance) -> None:
        # Asked before a sound is made. A filter that subscribes to an event
        # has already lost the race with the speaker.
        if not self.guard(utterance.text, utterance.language, utterance.session_id):
            return

        self.gate.on_speech_start()
        try:
            handle = self.tts.speak(utterance.text, utterance.language)
            handle.wait()
        except LomasError as exc:
            # No voice is not no lesson. A missing piper binary must not end
            # the class in front of the room; it makes the robot quiet, and
            # the log says why. Said once - a warning on every sentence is a
            # log nobody reads.
            if not self._mute_reported:
                self._mute_reported = True
                self.log.error("no voice, teaching silently: %s", exc)
        finally:
            self.gate.on_speech_end()

        who = f" [{utterance.student_name}]" if utterance.student_name else ""
        self.log.info("%s%s", utterance.text, who)
        self.bus.publish(ROBOT_SPOKE, utterance)

    def _on_pause(self, _event: str, _payload) -> None:
        self.tts.stop()

    def stop(self) -> None:
        self.tts.stop()
        self.gate.on_speech_end()
