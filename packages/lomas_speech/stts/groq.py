from __future__ import annotations

import os
from typing import Iterator

from lomas_core.errors import LomasError
from lomas_core.schema import SttConfig
from lomas_speech.stt import STT_ENGINES
from lomas_speech.types import Transcript

AUDIO_FIELD = "file"
UPLOAD_NAME = "utterance.wav"
WAV_MIME = "audio/wav"


@STT_ENGINES.register("groq")
class GroqStt:
    """Whisper through Groq's OpenAI-compatible transcription endpoint.

    Fast enough for classroom turn-taking, which is the whole reason for
    sending audio off the robot at all.
    """

    def __init__(self, cfg: SttConfig) -> None:
        self.cfg = cfg

    def _key(self) -> str:
        key = os.environ.get(self.cfg.api_key_env, "")
        if not key:
            raise LomasError(
                f"{self.cfg.api_key_env} is not set. Export it, or use "
                "speech.stt.engine: vosk for offline transcription."
            )
        return key

    def transcribe(self, audio: bytes, language: str = "") -> Transcript:
        import httpx

        language = language or self.cfg.language
        response = httpx.post(
            f"{self.cfg.api_base}/audio/transcriptions",
            headers={"Authorization": f"Bearer {self._key()}"},
            files={AUDIO_FIELD: (UPLOAD_NAME, audio, WAV_MIME)},
            data={"model": self.cfg.model, "language": language},
            timeout=self.cfg.timeout_seconds,
        )
        response.raise_for_status()
        return Transcript(text=response.json().get("text", "").strip(), language=language)

    def stream(self, language: str = "") -> Iterator[Transcript]:
        raise LomasError("groq transcription is request-response; capture then transcribe")
