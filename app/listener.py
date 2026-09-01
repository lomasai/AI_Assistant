from __future__ import annotations

from typing import Any

from lomas_core import logging as log
from lomas_core.clock import Clock
from lomas_core.contracts import (
    QUESTION_ASKED,
    ROBOT_STATE,
    QuestionAsked,
)
from lomas_core.errors import LomasError
from lomas_core.events import EventBus
from lomas_core.schema import Config
from lomas_speech.recorder import loudness

LISTENING = "listening"
IDLE = "idle"
SOURCE = "microphone"


class Listener:
    """A child's voice, turned into a question the rest of the system knows.

    Push to talk, not a wake word. In a room of forty children the robot
    cannot work out who spoke, and guessing wrong records one child's answer
    against another. The teacher taps a name and presses listen, so the
    attribution is a decision by the person who can see the room.

    It publishes `question.asked` and nothing else, which is the same event
    the teacher's typed question produces - so everything downstream already
    works.
    """

    def __init__(
        self,
        cfg: Config,
        bus: EventBus,
        clock: Clock,
        recorder: Any,
        stt: Any,
        gate: Any = None,
    ) -> None:
        self.cfg = cfg
        self.bus = bus
        self.clock = clock
        self.recorder = recorder
        self.stt = stt
        self.gate = gate
        self.log = log.get("listen")
        self.heard = 0

    @property
    def available(self) -> bool:
        return bool(self.recorder is not None and self.recorder.available)

    def describe(self) -> str:
        return self.recorder.describe() if self.recorder else "none"

    def listen(
        self,
        session_id: str = "",
        student_id: str = "",
        student_name: str = "",
        seconds: float = 0.0,
        language: str = "",
    ) -> dict:
        """Record, transcribe, publish. Blocking, because the caller is a web
        request and the teacher is standing there waiting for it."""
        if not self.available:
            raise LomasError(
                "no microphone. On Raspberry Pi OS arecord is already there; "
                "check speech.audio.recorder and the device."
            )

        audio = self._capture(session_id, seconds or self.cfg.speech.audio.record_seconds)
        if not audio:
            return {"text": "", "reason": "nothing was recorded"}

        # Checked before the network, not after. Whisper hallucinates on
        # silence, and a sentence nobody said, attributed to a named child,
        # is worse than no answer at all.
        peak, rms = loudness(audio)
        if peak < self.cfg.speech.audio.silence_peak:
            self.log.info("too quiet to send: peak %.3f", peak)
            return {"text": "", "reason": "nothing was heard", "peak": round(peak, 3)}

        language = language or self.cfg.content.language
        try:
            heard = self.stt.transcribe(audio, language)
        except LomasError:
            raise
        except Exception as exc:
            raise LomasError(f"could not transcribe: {exc}") from exc

        spoken = heard.text.strip()
        if len(_letters(spoken)) < self.cfg.speech.stt.min_characters:
            # "." and "So, let's go." are what whisper returns for a room that
            # said nothing. They are not questions and must not become ones.
            self.log.info("discarded as noise: %r", spoken)
            return {"text": "", "reason": "nothing was said", "discarded": spoken}

        self.heard += 1
        self.log.info("heard: %s", spoken)
        self.bus.publish(
            QUESTION_ASKED,
            QuestionAsked(
                session_id=session_id,
                text=spoken,
                student_id=student_id,
                student_name=student_name,
            ),
        )
        return {"text": spoken, "language": heard.language, "student_id": student_id,
                "peak": round(peak, 3)}

    def _capture(self, session_id: str, seconds: float) -> bytes:
        audio = self.cfg.speech.audio

        # The face widens its eyes and shows that it is hearing you. It is the
        # most reassuring screen in the product, and it costs one event.
        self._state(session_id, LISTENING, seconds)
        try:
            return self.recorder.record(seconds, audio.sample_rate)
        finally:
            self._state(session_id, IDLE, 0.0)

    def _state(self, session_id: str, state: str, seconds: float) -> None:
        self.bus.publish(
            ROBOT_STATE,
            {"session_id": session_id, "state": state, "by": SOURCE, "seconds": seconds},
        )


def _letters(text: str) -> str:
    """Punctuation is not speech. A transcript of "." carries no letters."""
    return "".join(c for c in text if c.isalnum())
