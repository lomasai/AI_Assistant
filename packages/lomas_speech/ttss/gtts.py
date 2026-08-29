from __future__ import annotations

from pathlib import Path

from lomas_core.errors import LomasError
from lomas_core.schema import TtsConfig
from lomas_speech.player import Player
from lomas_speech.tts import TTS_ENGINES
from lomas_speech.types import SpeechHandle

SILENT = 0.0


@TTS_ENGINES.register("gtts")
class GttsTts:
    """Cloud fallback. Better voices than piper in some languages, useless the
    moment the school's internet drops."""

    def __init__(self, cfg: TtsConfig) -> None:
        self.cfg = cfg
        self.player = Player(cfg.player, cfg.player_command)
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
        # Saved and then played. The file on its own is a robot that mimes.
        gTTS(text=text, lang=language).save(self.cfg.scratch_file)
        self._handle = handle
        try:
            self.player.play_file(Path(self.cfg.scratch_file))
        finally:
            handle.finish()
        return handle

    def stop(self) -> None:
        self.player.stop()
        if self._handle is not None and not self._handle.done:
            self._handle.cancel()

    def amplitude(self) -> float:
        return SILENT
