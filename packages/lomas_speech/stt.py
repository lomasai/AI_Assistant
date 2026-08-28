from __future__ import annotations

from typing import Iterator, Protocol, runtime_checkable

from lomas_core.registry import Registry
from lomas_speech.types import Transcript


@runtime_checkable
class SpeechToText(Protocol):
    def transcribe(self, audio: bytes, language: str) -> Transcript: ...

    def stream(self, language: str) -> Iterator[Transcript]: ...


STT_ENGINES: Registry[SpeechToText] = Registry("speech-to-text engine")
