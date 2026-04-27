"""Speech-to-Text module for server-side audio transcription.

This module supports:
- File-based audio transcription
- Byte stream transcription
- Provider routing: local Whisper, Whisper API, or placeholder
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Iterable

import httpx


class STTError(Exception):
    """Raised when transcription fails."""


@dataclass(slots=True)
class STTConfig:
    """Configuration for the STT service."""

    provider: str = "placeholder"  # whisper_local | whisper_api | groq_whisper | placeholder
    whisper_model: str = "base"
    whisper_api_url: str = "https://api.openai.com/v1/audio/transcriptions"
    whisper_api_key: str = ""
    whisper_api_model_name: str = "whisper-1"
    groq_api_url: str = "https://api.groq.com/openai/v1/audio/transcriptions"
    groq_api_key: str = ""
    groq_model_name: str = "whisper-large-v3-turbo"
    language: str = ""
    request_timeout_s: float = 30.0
    placeholder_text: str = "Transcription placeholder output."

    @classmethod
    def from_env(cls) -> "STTConfig":
        """Build config from environment variables."""
        return cls(
            provider=os.getenv("STT_PROVIDER", "placeholder").strip().lower(),
            whisper_model=os.getenv("WHISPER_MODEL", "base").strip(),
            whisper_api_url=os.getenv(
                "WHISPER_API_URL",
                "https://api.openai.com/v1/audio/transcriptions",
            ).strip(),
            whisper_api_key=os.getenv("WHISPER_API_KEY", "").strip(),
            whisper_api_model_name=os.getenv("WHISPER_API_MODEL", "whisper-1").strip(),
            groq_api_url=os.getenv(
                "GROQ_WHISPER_API_URL",
                "https://api.groq.com/openai/v1/audio/transcriptions",
            ).strip(),
            groq_api_key=os.getenv("GROQ_API_KEY", "").strip(),
            groq_model_name=os.getenv("GROQ_WHISPER_MODEL", "whisper-large-v3-turbo").strip(),
            language=os.getenv("STT_LANGUAGE", "").strip(),
            request_timeout_s=float(os.getenv("STT_TIMEOUT_SECONDS", "30")),
            placeholder_text=os.getenv(
                "STT_PLACEHOLDER_TEXT",
                "Transcription placeholder output.",
            ).strip(),
        )


class STTService:
    """Production-friendly STT service with async interfaces."""

    def __init__(self, config: STTConfig | None = None, http_client: httpx.AsyncClient | None = None) -> None:
        self.config = config or STTConfig.from_env()
        self._http_client = http_client or httpx.AsyncClient(timeout=self.config.request_timeout_s)
        self._owns_http_client = http_client is None
        self._whisper_model = None
        self._whisper_lock = asyncio.Lock()

    async def close(self) -> None:
        """Close network resources owned by this service."""
        if self._owns_http_client:
            await self._http_client.aclose()

    async def transcribe_file(self, audio_path: str | Path) -> str:
        """Transcribe a local audio file and return cleaned text."""
        path = Path(audio_path)
        if not path.exists():
            raise STTError(f"Audio file not found: {path}")

        audio_bytes = await asyncio.to_thread(path.read_bytes)
        return await self.transcribe_bytes(audio_bytes=audio_bytes, filename=path.name)

    async def transcribe_bytes(self, audio_bytes: bytes, filename: str = "audio.wav") -> str:
        """Transcribe in-memory audio bytes and return cleaned text."""
        if not audio_bytes:
            raise STTError("Audio input is empty.")

        if self.config.provider == "whisper_local":
            text = await self._transcribe_with_local_whisper(audio_bytes=audio_bytes, filename=filename)
        elif self.config.provider == "whisper_api":
            text = await self._transcribe_with_whisper_api(audio_bytes=audio_bytes, filename=filename)
        elif self.config.provider == "groq_whisper":
            text = await self._transcribe_with_groq_whisper(audio_bytes=audio_bytes, filename=filename)
        elif self.config.provider == "placeholder":
            text = await self._transcribe_with_placeholder()
        else:
            raise STTError(f"Unsupported STT provider: {self.config.provider}")

        return self._clean_text(text)

    async def transcribe_stream(
        self,
        stream: AsyncIterator[bytes] | Iterable[bytes],
        filename: str = "stream.wav",
    ) -> str:
        """Transcribe audio chunks from an async or sync stream."""
        chunks: list[bytes] = []

        if hasattr(stream, "__aiter__"):
            async for chunk in stream:  # type: ignore[misc]
                if chunk:
                    chunks.append(chunk)
        else:
            for chunk in stream:
                if chunk:
                    chunks.append(chunk)

        if not chunks:
            raise STTError("No audio data received from stream.")

        return await self.transcribe_bytes(audio_bytes=b"".join(chunks), filename=filename)

    async def _transcribe_with_placeholder(self) -> str:
        """Return a deterministic placeholder response for development."""
        return self.config.placeholder_text

    async def _transcribe_with_whisper_api(self, audio_bytes: bytes, filename: str) -> str:
        """Transcribe using an HTTP Whisper-compatible API endpoint."""
        headers = {}
        if self.config.whisper_api_key:
            headers["Authorization"] = f"Bearer {self.config.whisper_api_key}"

        data = {"model": self.config.whisper_api_model_name}
        if self.config.language:
            data["language"] = self.config.language

        files = {"file": (filename, audio_bytes, "audio/wav")}

        try:
            response = await self._http_client.post(
                self.config.whisper_api_url,
                headers=headers,
                data=data,
                files=files,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise STTError(f"Whisper API request failed: {exc}") from exc

        payload = response.json()
        text = payload.get("text") if isinstance(payload, dict) else None
        if not text:
            raise STTError("Whisper API returned no transcription text.")

        return str(text)

    async def _transcribe_with_groq_whisper(self, audio_bytes: bytes, filename: str) -> str:
        """Transcribe using Groq's OpenAI-compatible Whisper endpoint."""
        if not self.config.groq_api_key:
            raise STTError("GROQ_API_KEY is required when STT_PROVIDER=groq_whisper.")

        headers = {"Authorization": f"Bearer {self.config.groq_api_key}"}
        data = {
            "model": self.config.groq_model_name,
            "response_format": "json",
            "temperature": "0",
        }
        if self.config.language:
            data["language"] = self.config.language

        files = {"file": (filename, audio_bytes, self._content_type_for_filename(filename))}

        try:
            response = await self._http_client.post(
                self.config.groq_api_url,
                headers=headers,
                data=data,
                files=files,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise STTError(f"Groq Whisper HTTP error {exc.response.status_code}: {exc.response.text}") from exc
        except httpx.HTTPError as exc:
            raise STTError(f"Groq Whisper request failed: {exc}") from exc

        payload = response.json()
        text = payload.get("text") if isinstance(payload, dict) else None
        if not text:
            raise STTError("Groq Whisper returned no transcription text.")
        return str(text)

    async def _transcribe_with_local_whisper(self, audio_bytes: bytes, filename: str) -> str:
        """Transcribe using the local open-source Whisper package."""
        model = await self._get_or_load_whisper_model()

        suffix = Path(filename).suffix or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(audio_bytes)
            temp_path = tmp.name

        try:
            result = await asyncio.to_thread(
                model.transcribe,
                temp_path,
                language=self.config.language or None,
                fp16=False,
            )
        except Exception as exc:  # noqa: BLE001
            raise STTError(f"Local Whisper transcription failed: {exc}") from exc
        finally:
            Path(temp_path).unlink(missing_ok=True)

        text = result.get("text") if isinstance(result, dict) else None
        if not text:
            raise STTError("Local Whisper returned no transcription text.")

        return str(text)

    async def _get_or_load_whisper_model(self):
        """Load and cache local Whisper model once."""
        if self._whisper_model is not None:
            return self._whisper_model

        async with self._whisper_lock:
            if self._whisper_model is not None:
                return self._whisper_model

            try:
                import whisper  # type: ignore
            except ImportError as exc:
                raise STTError(
                    "Local Whisper is not installed. "
                    "Install `openai-whisper` or use STT_PROVIDER=whisper_api/placeholder."
                ) from exc

            self._whisper_model = await asyncio.to_thread(whisper.load_model, self.config.whisper_model)
            return self._whisper_model

    @staticmethod
    def _clean_text(text: str) -> str:
        """Normalize whitespace and trim boundary punctuation."""
        cleaned = " ".join(text.replace("\n", " ").replace("\t", " ").split())
        return cleaned.strip(" \"'")

    @staticmethod
    def _content_type_for_filename(filename: str) -> str:
        suffix = Path(filename).suffix.lower()
        if suffix == ".webm":
            return "audio/webm"
        if suffix == ".mp3":
            return "audio/mpeg"
        if suffix == ".m4a":
            return "audio/mp4"
        if suffix == ".ogg":
            return "audio/ogg"
        return "audio/wav"


stt_service = STTService()
