from __future__ import annotations

from lomas_core.schema import TtsConfig
from lomas_speech.tts import TTS_ENGINES
from lomas_speech.types import SpeechHandle

SILENT = 0.0


@TTS_ENGINES.register("null")
class NullTts:
    """Says nothing and records everything.

    Tests assert on `spoken` rather than on audio, which is why the whole
    lesson flow can be verified with no speaker attached.
    """

    def __init__(self, cfg: TtsConfig | None = None) -> None:
        self.cfg = cfg
        self.spoken: list[tuple[str, str]] = []
        self._current: SpeechHandle | None = None

    def speak(self, text: str, language: str = "") -> SpeechHandle:
        self.spoken.append((text, language))
        handle = SpeechHandle(text=text, language=language)
        handle.finish()
        self._current = handle
        return handle

    def stop(self) -> None:
        if self._current is not None and not self._current.done:
            self._current.cancel()

    def amplitude(self) -> float:
        return SILENT

    def last(self) -> str:
        return self.spoken[-1][0] if self.spoken else ""
