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
from lomas_speech.recorder import Endpoint, loudness

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
        speakers: Any = None,
        voice: Any = None,
    ) -> None:
        self.cfg = cfg
        self.bus = bus
        self.clock = clock
        self.recorder = recorder
        self.stt = stt
        self.gate = gate
        # Who spoke, worked out rather than tapped. None keeps the old
        # behaviour: whoever the caller named.
        self.speakers = speakers
        # Only to ask whether it is still talking.
        self.voice = voice
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
        as_question: bool = True,
        attribute: bool = True,
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

        # A topic has no speaker worth working out, and the robot asking
        # "who was that?" about the subject of the lesson is nonsense.
        student_id, student_name, spoken, how = self._who(
            spoken, student_id, student_name, session_id, language) if attribute else (
            student_id, student_name, spoken, "not asked")

        self.heard += 1
        self.log.info("heard: %s%s", spoken, f" [{student_name}]" if student_name else "")
        if not as_question:
            # An answer, published by whoever asked for one. As a question as
            # well, the tutor explained the answer back to the child while
            # the quiz moved on, and the two talked over each other.
            return {"text": spoken, "language": heard.language, "student_id": student_id,
                    "student_name": student_name, "how": how, "peak": round(peak, 3)}
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
                "student_name": student_name, "how": how, "peak": round(peak, 3)}

    def _who(self, spoken: str, student_id: str, student_name: str,
             session_id: str, language: str) -> tuple[str, str, str, str]:
        """Who said it. The tapped name goes in as one more opinion, and
        comes back out first - `tapped` is the head of the chain."""
        if self.speakers is None:
            return student_id, student_name, spoken, "caller"

        found = self.speakers.resolve(
            spoken, tapped=(student_id, student_name), session_id=session_id, language=language)
        return found.student_id, found.name, found.text or spoken, found.how

    def _capture(self, session_id: str, seconds: float) -> bytes:
        audio = self.cfg.speech.audio
        self._wait_for_robot()

        # The face widens its eyes and shows that it is hearing you. It is the
        # most reassuring screen in the product, and it costs one event.
        self._state(session_id, LISTENING, seconds)
        try:
            return self.recorder.record(seconds, audio.sample_rate, endpoint=Endpoint(
                silence_ms=audio.stop_after_silence_ms,
                no_speech_seconds=audio.no_speech_seconds,
                chunk_ms=audio.chunk_ms,
                speech_fraction=audio.speech_fraction,
                min_gap_rms=audio.min_gap_rms,
                min_gap_ratio=audio.min_gap_ratio,
                min_rms=audio.min_rms,
                smooth_samples=audio.smooth_samples,
            ))
        finally:
            # How the turn ended and how loud the room was, in the trace. The
            # last run could only say "fifteen seconds, every time".
            turn = getattr(self.recorder, "last_turn", None)
            heard = turn.summary() if turn is not None else {}
            if heard:
                self.log.info("recorded %.1fs, stopped: %s", heard["seconds"], heard["stopped"])
            self._state(session_id, IDLE, 0.0, heard)

    def _wait_for_robot(self) -> None:
        """Start hearing once the robot has stopped talking.

        Pressed mid-sentence, the recording carried the robot's own next
        question into a child's answer. Bounded, because a voice that never
        ends must not leave the teacher's button hanging.
        """
        audio = self.cfg.speech.audio
        deadline = self.clock.now() + audio.wait_for_robot_seconds
        while self._talking() and self.clock.now() < deadline:
            self.clock.sleep(audio.quiet_poll_seconds)

    def _talking(self) -> bool:
        if self.voice is not None and self.voice.busy:
            return True
        return self.gate is not None and self.gate.is_muted()

    def _state(self, session_id: str, state: str, seconds: float, heard: dict | None = None) -> None:
        self.bus.publish(
            ROBOT_STATE,
            {"session_id": session_id, "state": state, "by": SOURCE, "seconds": seconds,
             **(heard or {})},
        )


def _letters(text: str) -> str:
    """Punctuation is not speech. A transcript of "." carries no letters."""
    return "".join(c for c in text if c.isalnum())
