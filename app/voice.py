from __future__ import annotations

from lomas_core import logging as log
from lomas_core.contracts import ROBOT_SAY, ROBOT_SPOKE, Utterance
from lomas_core.events import EventBus
from lomas_speech import DuplexGate, TextToSpeech


class Voice:
    """Turns `robot.say` into sound, and mutes the microphones while it does.

    Steps publish; this is the only thing in the system holding a speaker.
    Publishing is synchronous, so a step that says something has finished
    saying it when `say` returns.
    """

    def __init__(self, tts: TextToSpeech, gate: DuplexGate, bus: EventBus) -> None:
        self.tts = tts
        self.gate = gate
        self.bus = bus
        self.log = log.get("voice")
        bus.subscribe(ROBOT_SAY, self._on_say)

    def _on_say(self, _event: str, utterance: Utterance) -> None:
        self.gate.on_speech_start()
        try:
            handle = self.tts.speak(utterance.text, utterance.language)
            handle.wait()
        finally:
            self.gate.on_speech_end()

        who = f" [{utterance.student_name}]" if utterance.student_name else ""
        self.log.info("%s%s", utterance.text, who)
        self.bus.publish(ROBOT_SPOKE, utterance)

    def stop(self) -> None:
        self.tts.stop()
        self.gate.on_speech_end()
