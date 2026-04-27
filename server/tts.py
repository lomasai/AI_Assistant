"""Text-to-Speech module for server-side speech generation.

This module supports:
- Text input to WAV audio bytes
- Audio output to file
- Chunked audio streaming
- Pluggable providers: local Coqui TTS, HTTP API, placeholder
"""

from __future__ import annotations

import asyncio
import base64
import os
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator

import httpx


class TTSError(Exception):
    """Raised when text-to-speech synthesis fails."""


@dataclass(slots=True)
class TTSConfig:
    """Configuration for the TTS service."""

    provider: str = "placeholder"  # coqui_local | api | placeholder
    sample_rate: int = 22050
    voice: str = "default"
    api_url: str = ""
    api_key: str = ""
    api_timeout_s: float = 30.0
    coqui_model_name: str = "tts_models/en/ljspeech/tacotron2-DDC"

    @classmethod
    def from_env(cls) -> "TTSConfig":
        """Build config from environment variables."""
        return cls(
            provider=os.getenv("TTS_PROVIDER", "placeholder").strip().lower(),
            sample_rate=int(os.getenv("TTS_SAMPLE_RATE", "22050")),
            voice=os.getenv("TTS_VOICE", "default").strip(),
            api_url=os.getenv("TTS_API_URL", "").strip(),
            api_key=os.getenv("TTS_API_KEY", "").strip(),
            api_timeout_s=float(os.getenv("TTS_API_TIMEOUT_SECONDS", "30")),
            coqui_model_name=os.getenv(
                "COQUI_MODEL_NAME",
                "tts_models/en/ljspeech/tacotron2-DDC",
            ).strip(),
        )


class TTSService:
    """Production-friendly TTS service with async interfaces."""

    def __init__(self, config: TTSConfig | None = None, http_client: httpx.AsyncClient | None = None) -> None:
        self.config = config or TTSConfig.from_env()
        self._http_client = http_client or httpx.AsyncClient(timeout=self.config.api_timeout_s)
        self._owns_http_client = http_client is None
        self._coqui_model = None
        self._coqui_lock = asyncio.Lock()

    async def close(self) -> None:
        """Close network resources owned by this service."""
        if self._owns_http_client:
            await self._http_client.aclose()

    async def synthesize_to_file(self, text: str, output_path: str | Path) -> Path:
        """Generate speech audio for text and persist it to disk."""
        audio_bytes = await self.synthesize_to_bytes(text=text)
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(path.write_bytes, audio_bytes)
        return path

    async def synthesize_to_bytes(self, text: str) -> bytes:
        """Generate speech audio for text and return WAV bytes."""
        clean_text = self._normalize_text(text)
        if not clean_text:
            raise TTSError("TTS input text is empty.")

        if self.config.provider == "coqui_local":
            return await self._synthesize_with_coqui(clean_text)
        if self.config.provider == "api":
            return await self._synthesize_with_api(clean_text)
        if self.config.provider == "placeholder":
            return await self._synthesize_placeholder(clean_text)

        raise TTSError(f"Unsupported TTS provider: {self.config.provider}")

    async def synthesize_stream(self, text: str, chunk_size: int = 4096) -> AsyncIterator[bytes]:
        """Yield synthesized audio in chunks for streaming responses."""
        audio = await self.synthesize_to_bytes(text=text)
        for index in range(0, len(audio), max(1, chunk_size)):
            yield audio[index : index + chunk_size]

    async def _synthesize_with_api(self, text: str) -> bytes:
        """Generate speech through an external HTTP API backend."""
        if not self.config.api_url:
            raise TTSError("TTS_API_URL is required for provider=api.")

        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        payload = {"text": text, "voice": self.config.voice}

        try:
            response = await self._http_client.post(self.config.api_url, json=payload, headers=headers)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise TTSError(f"TTS API request failed: {exc}") from exc

        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type:
            data = response.json()
            audio_b64 = data.get("audio_base64") if isinstance(data, dict) else None
            if not audio_b64:
                raise TTSError("TTS API JSON response missing 'audio_base64'.")
            try:
                return base64.b64decode(audio_b64)
            except Exception as exc:  # noqa: BLE001
                raise TTSError(f"Invalid base64 audio from API: {exc}") from exc

        if not response.content:
            raise TTSError("TTS API returned empty audio content.")
        return response.content

    async def _synthesize_with_coqui(self, text: str) -> bytes:
        """Generate speech via local Coqui TTS model."""
        model = await self._get_or_load_coqui_model()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            temp_path = Path(tmp.name)

        try:
            await asyncio.to_thread(model.tts_to_file, text=text, file_path=str(temp_path))
            audio = await asyncio.to_thread(temp_path.read_bytes)
        except Exception as exc:  # noqa: BLE001
            raise TTSError(f"Local Coqui synthesis failed: {exc}") from exc
        finally:
            temp_path.unlink(missing_ok=True)

        if not audio:
            raise TTSError("Local Coqui returned empty audio.")
        return audio

    async def _synthesize_placeholder(self, text: str) -> bytes:
        """Generate simple silent WAV with duration based on text length."""
        approx_seconds = max(1, min(5, len(text) // 25 + 1))
        frame_count = self.config.sample_rate * approx_seconds
        pcm_silence = b"\x00\x00" * frame_count
        return await asyncio.to_thread(self._build_wav_bytes, pcm_silence, self.config.sample_rate)

    async def _get_or_load_coqui_model(self):
        """Load and cache Coqui TTS model once."""
        if self._coqui_model is not None:
            return self._coqui_model

        async with self._coqui_lock:
            if self._coqui_model is not None:
                return self._coqui_model
            try:
                from TTS.api import TTS  # type: ignore
            except ImportError as exc:
                raise TTSError(
                    "Coqui TTS is not installed. Install `TTS` or use TTS_PROVIDER=api/placeholder."
                ) from exc

            self._coqui_model = await asyncio.to_thread(TTS, model_name=self.config.coqui_model_name)
            return self._coqui_model

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize whitespace for cleaner speech synthesis."""
        return " ".join(text.replace("\n", " ").replace("\t", " ").split()).strip()

    @staticmethod
    def _build_wav_bytes(pcm_16le: bytes, sample_rate: int) -> bytes:
        """Wrap raw 16-bit mono PCM data into a WAV container."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            temp_path = Path(tmp.name)
        try:
            with wave.open(str(temp_path), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(pcm_16le)
            return temp_path.read_bytes()
        finally:
            temp_path.unlink(missing_ok=True)


tts_service = TTSService()
