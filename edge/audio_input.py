"""Raspberry Pi audio input module.

Responsibilities:
- Record audio from microphone
- Send audio to server STT endpoint
- Return transcribed text response
"""

from __future__ import annotations

import asyncio
import base64
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import httpx


class EdgeAudioError(Exception):
    """Raised when edge audio capture or STT communication fails."""


@dataclass(slots=True)
class EdgeAudioConfig:
    """Configuration for Pi audio capture and server communication."""

    server_base_url: str = "http://127.0.0.1:8000"
    stt_endpoint: str = "/stt"
    sample_rate: int = 16000
    channels: int = 1
    arecord_format: str = "S16_LE"
    input_device: str | None = None
    request_timeout_s: float = 60.0

    @classmethod
    def from_env(cls) -> "EdgeAudioConfig":
        """Build config from environment variables."""
        return cls(
            server_base_url=os.getenv("EDGE_SERVER_BASE_URL", "http://127.0.0.1:8000").strip(),
            stt_endpoint=os.getenv("EDGE_STT_ENDPOINT", "/stt").strip(),
            sample_rate=int(os.getenv("EDGE_AUDIO_SAMPLE_RATE", "16000")),
            channels=int(os.getenv("EDGE_AUDIO_CHANNELS", "1")),
            arecord_format=os.getenv("EDGE_AUDIO_ARECORD_FORMAT", "S16_LE").strip(),
            input_device=os.getenv("EDGE_AUDIO_INPUT_DEVICE", "").strip() or None,
            request_timeout_s=float(os.getenv("EDGE_AUDIO_TIMEOUT_SECONDS", "60")),
        )


class EdgeAudioInput:
    """Audio recorder + STT client for edge device."""

    def __init__(self, config: EdgeAudioConfig | None = None, http_client: httpx.AsyncClient | None = None) -> None:
        self.config = config or EdgeAudioConfig.from_env()
        self._http_client = http_client or httpx.AsyncClient(timeout=self.config.request_timeout_s)
        self._owns_http_client = http_client is None

    async def close(self) -> None:
        """Close owned HTTP resources."""
        if self._owns_http_client:
            await self._http_client.aclose()

    async def record_audio(self, duration_seconds: int, output_path: str | Path | None = None) -> Path:
        """Record audio from Pi microphone into a WAV file."""
        if duration_seconds <= 0:
            raise EdgeAudioError("duration_seconds must be greater than 0.")

        target = Path(output_path) if output_path else Path(tempfile.mkstemp(suffix=".wav")[1])
        target.parent.mkdir(parents=True, exist_ok=True)

        if shutil.which("arecord") is None:
            raise EdgeAudioError("`arecord` not found. Install ALSA utils on Raspberry Pi.")

        cmd = [
            "arecord",
            "-q",
            "-d",
            str(duration_seconds),
            "-f",
            self.config.arecord_format,
            "-r",
            str(self.config.sample_rate),
            "-c",
            str(self.config.channels),
            str(target),
        ]
        if self.config.input_device:
            cmd[1:1] = ["-D", self.config.input_device]

        try:
            await asyncio.to_thread(
                subprocess.run,
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            raise EdgeAudioError(f"Audio recording failed: {stderr or exc}") from exc

        if not target.exists() or target.stat().st_size == 0:
            raise EdgeAudioError("Recorded audio file is empty.")
        return target

    async def send_to_stt(self, audio_path: str | Path) -> str:
        """Send recorded WAV file to server STT endpoint and return text."""
        path = Path(audio_path)
        if not path.exists():
            raise EdgeAudioError(f"Audio file not found: {path}")

        audio_bytes = await asyncio.to_thread(path.read_bytes)
        payload = {
            "audio_base64": base64.b64encode(audio_bytes).decode("ascii"),
            "filename": path.name,
        }
        url = f"{self.config.server_base_url.rstrip('/')}/{self.config.stt_endpoint.lstrip('/')}"

        try:
            response = await self._http_client.post(url, json=payload)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text.strip()
            raise EdgeAudioError(
                f"STT endpoint error {exc.response.status_code}: {detail}"
            ) from exc
        except httpx.HTTPError as exc:
            raise EdgeAudioError(f"Failed to call STT endpoint: {exc}") from exc

        try:
            data = response.json()
        except ValueError as exc:
            raise EdgeAudioError("STT endpoint returned non-JSON response.") from exc

        text = data.get("text") if isinstance(data, dict) else None
        if not isinstance(text, str) or not text.strip():
            raise EdgeAudioError("STT response did not contain valid text.")
        return text.strip()

    async def capture_and_transcribe(self, duration_seconds: int = 4) -> str:
        """Record audio and return transcription from server."""
        wav_path = await self.record_audio(duration_seconds=duration_seconds)
        try:
            return await self.send_to_stt(wav_path)
        finally:
            wav_path.unlink(missing_ok=True)


edge_audio_input = EdgeAudioInput()
