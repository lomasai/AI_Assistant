from __future__ import annotations

from lomas_core.errors import LomasError
from lomas_core.schema import TtsConfig
from lomas_speech.tts import TTS_ENGINES
from lomas_speech.types import SpeechHandle

SILENT = 0.0


@TTS_ENGINES.register("gtts")
class GttsTts:
    """Cloud fallback. Better voices than piper in some languages, useless the
    moment the school's internet drops."""

    def __init__(self, cfg: TtsConfig) -> None:
        self.cfg = cfg
        self._handle: SpeechHandle | None = None

    def speak(self, text: str, language: str = "") -> SpeechHandle:
        try:
            from gtts import gTTS
        except ImportError as exc:
            raise LomasError(
                "gtts is not installed. pip install gTTS, or use "
                "speech.tts.engine: piper."
            ) from exc

        language = language or self.cfg.fallback_language
        handle = SpeechHandle(text=text, language=language)
        gTTS(text=text, lang=language).save(self.cfg.scratch_file)
        handle.finish()
        self._handle = handle
        return handle

    def stop(self) -> None:
        if self._handle is not None and not self._handle.done:
            self._handle.cancel()

    def amplitude(self) -> float:
        return SILENT
