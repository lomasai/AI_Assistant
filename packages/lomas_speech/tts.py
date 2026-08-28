from __future__ import annotations

from typing import Protocol, runtime_checkable

from lomas_core.registry import Registry
from lomas_speech.types import SpeechHandle


@runtime_checkable
class TextToSpeech(Protocol):
    def speak(self, text: str, language: str) -> SpeechHandle: ...

    def stop(self) -> None: ...

    def amplitude(self) -> float: ...


TTS_ENGINES: Registry[TextToSpeech] = Registry("text-to-speech engine")
