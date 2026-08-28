from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from lomas_core.errors import LomasError
from lomas_core.schema import SttConfig
from lomas_speech.stt import STT_ENGINES
from lomas_speech.types import Transcript


@STT_ENGINES.register("vosk")
class VoskStt:
    """Offline transcription. Slower and less accurate than Whisper, and the
    reason a government school with no internet still gets a working robot."""

    def __init__(self, cfg: SttConfig) -> None:
        self.cfg = cfg
        self._models: dict[str, object] = {}

    def _model_for(self, language: str):
        if language in self._models:
            return self._models[language]
        try:
            from vosk import Model
        except ImportError as exc:
            raise LomasError(
                "vosk is not installed. pip install vosk, or use "
                "speech.stt.engine: groq."
            ) from exc

        path = Path(self.cfg.model_path) / language
        if not path.exists():
            raise LomasError(
                f"no Vosk model for '{language}' at {path}. Download the small "
                "model for that language and unpack it there."
            )
        self._models[language] = Model(str(path))
        return self._models[language]

    def transcribe(self, audio: bytes, language: str = "") -> Transcript:
        from vosk import KaldiRecognizer

        language = language or self.cfg.language
        recogniser = KaldiRecognizer(self._model_for(language), self.cfg.sample_rate)
        recogniser.AcceptWaveform(audio)
        result = json.loads(recogniser.FinalResult())
        return Transcript(text=result.get("text", "").strip(), language=language)

    def stream(self, language: str = "") -> Iterator[Transcript]:
        raise LomasError("streaming vosk needs an audio source; use transcribe()")
