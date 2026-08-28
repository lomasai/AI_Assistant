from __future__ import annotations

from collections import deque
from typing import Iterator

from lomas_core.schema import SttConfig
from lomas_speech.stt import STT_ENGINES
from lomas_speech.types import Transcript


@STT_ENGINES.register("keyboard")
class KeyboardStt:
    """Typed input standing in for a microphone.

    A test or the debug console queues what was 'said' and the rest of the
    system cannot tell the difference.
    """

    def __init__(self, cfg: SttConfig) -> None:
        self.cfg = cfg
        self._queued: deque[str] = deque()

    def say(self, text: str) -> None:
        self._queued.append(text)

    def transcribe(self, audio: bytes | None = None, language: str = "") -> Transcript:
        text = self._queued.popleft() if self._queued else ""
        return Transcript(text=text, language=language or self.cfg.language)

    def stream(self, language: str = "") -> Iterator[Transcript]:
        while self._queued:
            yield self.transcribe(language=language)
