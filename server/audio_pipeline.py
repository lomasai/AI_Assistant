"""Backend-owned audio pipeline, wake word, STT/TTS and turn-taking."""

from __future__ import annotations

import asyncio
import base64
import time
import wave
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Literal
from uuid import uuid4

from server.config import AudioConfig, PROJECT_ROOT, STTRuntimeConfig, TTSRuntimeConfig, VADConfig, WakeWordConfig
from server.interfaces import AudioChunk
from server.stt import STTError, stt_service
from server.teaching import InvalidTransitionError, StudentResponse, TeachingError, TeachingOrchestrator
from server.tts import TTSError, tts_service


AudioState = Literal[
    "disabled",
    "stopped",
    "ready",
    "listening",
    "speech_detected",
    "processing",
    "speaking",
    "timeout",
    "error",
]


class AudioPipelineError(Exception):
    """Raised for controlled audio pipeline failures."""


@dataclass(slots=True)
class AudioStatus:
    state: AudioState
    input_provider: str
    output_provider: str
    wake_word_provider: str
    stt_provider: str
    tts_provider: str
    microphone_available: bool
    speaker_available: bool
    speaking: bool = False
    listening: bool = False
    queue_depth: int = 0
    error: str | None = None

    def safe_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "input_provider": self.input_provider,
            "output_provider": self.output_provider,
            "wake_word_provider": self.wake_word_provider,
            "stt_provider": self.stt_provider,
            "tts_provider": self.tts_provider,
            "microphone_available": self.microphone_available,
            "speaker_available": self.speaker_available,
            "speaking": self.speaking,
            "listening": self.listening,
            "queue_depth": self.queue_depth,
            "error": self.error,
        }


class BoundedAudioBuffer:
    """Drop-oldest bounded buffer for audio chunks."""

    def __init__(self, max_chunks: int) -> None:
        self._chunks: deque[AudioChunk] = deque(maxlen=max(1, max_chunks))
        self.dropped_chunks = 0

    def append(self, chunk: AudioChunk) -> None:
        if len(self._chunks) == self._chunks.maxlen:
            self.dropped_chunks += 1
        self._chunks.append(chunk)

    def drain(self) -> list[AudioChunk]:
        chunks = list(self._chunks)
        self._chunks.clear()
        return chunks

    def __len__(self) -> int:
        return len(self._chunks)


class MockMicrophoneDriver:
    """Deterministic microphone for tests."""

    def __init__(self, config: AudioConfig, chunks: list[bytes] | None = None) -> None:
        self.config = config
        self.started = False
        self.unavailable = False
        self._chunks = chunks or [b"\x00" * 64, *([b"\x01" * 512] * 10), b"\x00" * 128]

    async def start(self) -> None:
        if self.unavailable:
            raise AudioPipelineError("Microphone unavailable.")
        self.started = True

    async def stop(self) -> None:
        self.started = False

    async def listen(self) -> AsyncIterator[AudioChunk]:
        sequence = 0
        while self.started:
            for payload in self._chunks:
                if not self.started:
                    return
                sequence += 1
                yield AudioChunk(
                    data=payload,
                    sample_rate=self.config.sample_rate,
                    channels=self.config.channels,
                    timestamp_utc=utc_now(),
                    sequence=sequence,
                    speech_hint=any(payload),
                )
                await asyncio.sleep(0)
            return


class LocalMicrophoneDriver(MockMicrophoneDriver):
    """Placeholder local microphone owner using optional sounddevice dependency."""

    async def start(self) -> None:
        try:
            import sounddevice  # noqa: F401, PLC0415
        except Exception as exc:  # noqa: BLE001
            raise AudioPipelineError("Local microphone provider is unavailable.") from exc
        await super().start()


class MockSpeakerDriver:
    def __init__(self) -> None:
        self.started = False
        self.spoken: list[str] = []
        self.cancelled = False

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.started = False

    async def speak(self, text: str, audio: bytes) -> None:
        _ = audio
        if not self.started:
            raise AudioPipelineError("Speaker unavailable.")
        self.cancelled = False
        self.spoken.append(text)
        await asyncio.sleep(0)

    async def cancel(self) -> None:
        self.cancelled = True


class LocalSpeakerDriver(MockSpeakerDriver):
    async def start(self) -> None:
        try:
            import sounddevice  # noqa: F401, PLC0415
        except Exception as exc:  # noqa: BLE001
            raise AudioPipelineError("Local speaker provider is unavailable.") from exc
        await super().start()


class MockWakeWordDetector:
    def __init__(self, config: WakeWordConfig) -> None:
        self.config = config
        self.last_activation = 0.0
        self.force_false = False

    async def initialize(self) -> None:
        return

    async def detect(self, chunk: AudioChunk) -> bool:
        if self.force_false or self.config.provider == "disabled":
            return False
        now = time.monotonic()
        if now - self.last_activation < self.config.cooldown_seconds:
            return False
        if chunk.speech_hint or b"wake" in chunk.data.lower():
            self.last_activation = now
            return True
        return False

    def health(self) -> dict[str, Any]:
        return {"provider": self.config.provider, "ready": self.config.provider != "disabled"}


class OpenWakeWordDetector(MockWakeWordDetector):
    async def initialize(self) -> None:
        model = resolve_path(self.config.model_path)
        if not model.exists():
            raise AudioPipelineError("Configured wake-word model is unavailable.")
        try:
            import openwakeword  # noqa: F401, PLC0415
        except Exception as exc:  # noqa: BLE001
            raise AudioPipelineError("openWakeWord is unavailable.") from exc


class VoiceActivityDetector:
    def __init__(self, config: VADConfig) -> None:
        self.config = config

    def classify(self, chunk: AudioChunk) -> bool:
        if chunk.speech_hint:
            return True
        if not chunk.data:
            return False
        active = sum(1 for byte in chunk.data if byte != 0)
        return active / len(chunk.data) >= self.config.speech_threshold


class MockSTTAdapter:
    def __init__(self, config: STTRuntimeConfig) -> None:
        self.config = config
        self.fail_next = False

    async def transcribe(self, audio_bytes: bytes) -> dict[str, Any]:
        if self.fail_next or self.config.provider == "disabled":
            self.fail_next = False
            return {"ok": False, "text": "", "status": "stt_failed"}
        if len(audio_bytes) < 1:
            return {"ok": False, "text": "", "status": "empty_audio"}
        return {"ok": True, "text": self.config.mock_transcript, "status": "ok"}


class LegacySTTAdapter(MockSTTAdapter):
    async def transcribe(self, audio_bytes: bytes) -> dict[str, Any]:
        try:
            return {"ok": True, "text": await stt_service.transcribe_bytes(audio_bytes, filename="voice.wav"), "status": "ok"}
        except STTError:
            return {"ok": False, "text": "", "status": "stt_failed"}


class WhisperCppSTTAdapter(MockSTTAdapter):
    async def transcribe(self, audio_bytes: bytes) -> dict[str, Any]:
        if not resolve_path(self.config.model_path).exists() or not resolve_path(self.config.executable_path).exists():
            return {"ok": False, "text": "", "status": "provider_unavailable"}
        return await super().transcribe(audio_bytes)


class MockTTSAdapter:
    def __init__(self, config: TTSRuntimeConfig) -> None:
        self.config = config
        self.fail_next = False

    async def synthesize(self, text: str) -> dict[str, Any]:
        if self.fail_next or self.config.provider == "disabled":
            self.fail_next = False
            return {"ok": False, "audio": b"", "status": "tts_failed"}
        return {"ok": True, "audio": build_silent_wav(400), "status": "ok"}


class LegacyTTSAdapter(MockTTSAdapter):
    async def synthesize(self, text: str) -> dict[str, Any]:
        try:
            return {"ok": True, "audio": await tts_service.synthesize_to_bytes(text), "status": "ok"}
        except TTSError:
            return {"ok": False, "audio": b"", "status": "tts_failed"}


class PiperTTSAdapter(MockTTSAdapter):
    async def synthesize(self, text: str) -> dict[str, Any]:
        if not resolve_path(self.config.model_path).exists() or not resolve_path(self.config.executable_path).exists():
            return {"ok": False, "audio": b"", "status": "provider_unavailable"}
        return await super().synthesize(text)


class AudioRuntime:
    """Single owner for microphone, speaker and voice teaching turns."""

    def __init__(
        self,
        audio_config: AudioConfig,
        wake_config: WakeWordConfig,
        vad_config: VADConfig,
        stt_config: STTRuntimeConfig,
        tts_config: TTSRuntimeConfig,
        teaching: TeachingOrchestrator,
    ) -> None:
        self.audio_config = audio_config
        self.wake_config = wake_config
        self.vad_config = vad_config
        self.stt_config = stt_config
        self.tts_config = tts_config
        self.teaching = teaching
        self.microphone = build_microphone(audio_config)
        self.speaker = build_speaker(audio_config)
        self.wake_word = build_wake_word(wake_config)
        self.vad = VoiceActivityDetector(vad_config)
        self.stt = build_stt(stt_config)
        self.tts = build_tts(tts_config)
        self.buffer = BoundedAudioBuffer(audio_config.buffer_max_chunks)
        self.state: AudioState = "stopped"
        self.error: str | None = None
        self.events: deque[dict[str, Any]] = deque(maxlen=200)
        self.speaking = False
        self.listening = False
        self._speech_queue: asyncio.Queue[tuple[str, bytes, str]] = asyncio.Queue(maxsize=8)
        self._speech_task: asyncio.Task | None = None

    async def start(self) -> None:
        if self.audio_config.input_provider == "disabled" and self.audio_config.output_provider == "disabled":
            self.state = "disabled"
            self._event("audio_state", {"state": self.state})
            return
        try:
            await self.microphone.start()
            await self.speaker.start()
            await self.wake_word.initialize()
            self.state = "ready"
            self.error = None
            self._speech_task = asyncio.create_task(self._speech_worker())
            self._event("audio_state", {"state": self.state})
        except AudioPipelineError as exc:
            self.state = "error"
            self.error = str(exc).split(":")[0]
            self._event("audio_state", {"state": self.state, "error": self.error})

    async def stop(self) -> None:
        await self.cancel()
        if self._speech_task:
            self._speech_task.cancel()
            try:
                await self._speech_task
            except asyncio.CancelledError:
                pass
            self._speech_task = None
        await self.microphone.stop()
        await self.speaker.stop()
        self.state = "stopped"
        self._event("audio_state", {"state": self.state})

    async def cancel(self) -> dict[str, Any]:
        await self.speaker.cancel()
        self.speaking = False
        self.listening = False
        self.buffer.drain()
        while not self._speech_queue.empty():
            try:
                self._speech_queue.get_nowait()
                self._speech_queue.task_done()
            except asyncio.QueueEmpty:
                break
        if self.state not in {"disabled", "error", "stopped"}:
            self.state = "ready"
        self._event("audio_cancelled", {"state": self.state})
        return self.status().safe_dict()

    def status(self) -> AudioStatus:
        return AudioStatus(
            state=self.state,
            input_provider=self.audio_config.input_provider,
            output_provider=self.audio_config.output_provider,
            wake_word_provider=self.wake_config.provider,
            stt_provider=self.stt_config.provider,
            tts_provider=self.tts_config.provider,
            microphone_available=self.state != "error" and self.audio_config.input_provider != "disabled",
            speaker_available=self.state != "error" and self.audio_config.output_provider != "disabled",
            speaking=self.speaking,
            listening=self.listening,
            queue_depth=self._speech_queue.qsize(),
            error=self.error,
        )

    async def wait_for_wake_word(self) -> dict[str, Any]:
        if self.wake_config.provider == "disabled":
            return {"activated": False, "status": "disabled"}
        started = time.monotonic()
        async for chunk in self.microphone.listen():
            if await self.wake_word.detect(chunk):
                self._event("wake_word", {"activated": True})
                return {"activated": True, "status": "ok"}
            if time.monotonic() - started > self.wake_config.activation_timeout_seconds:
                self.state = "timeout"
                self._event("wake_word", {"activated": False, "status": "timeout"})
                return {"activated": False, "status": "timeout"}
        return {"activated": False, "status": "no_audio"}

    async def push_to_talk(self, session_id: str | None = None) -> dict[str, Any]:
        if self.speaking and self.audio_config.half_duplex and not self.audio_config.barge_in_enabled:
            return {"ok": False, "status": "speaking"}
        recorded = await self.record_until_silence()
        if not recorded["ok"]:
            return recorded
        transcript = await self.stt.transcribe(recorded["audio"])
        self._event("stt_result", {"ok": transcript["ok"], "status": transcript["status"]})
        if not transcript["ok"]:
            self.state = "ready"
            return {"ok": False, "status": transcript["status"], "transcript": ""}
        if session_id:
            return await self.submit_voice_answer(session_id, transcript["text"])
        return {"ok": True, "status": "transcribed", "transcript": transcript["text"]}

    async def record_until_silence(self) -> dict[str, Any]:
        self.state = "listening"
        self.listening = True
        self._event("audio_state", {"state": self.state})
        started = time.monotonic()
        speech_started: float | None = None
        last_speech = 0.0
        async for chunk in self.microphone.listen():
            elapsed = time.monotonic() - started
            is_speech = self.vad.classify(chunk)
            self.buffer.append(chunk)
            if is_speech:
                if speech_started is None:
                    speech_started = time.monotonic()
                    self.state = "speech_detected"
                    self._event("audio_state", {"state": self.state})
                last_speech = time.monotonic()
            elif speech_started and (time.monotonic() - last_speech) * 1000 >= self.vad_config.silence_timeout_ms:
                break
            if elapsed >= min(self.audio_config.max_recording_seconds, self.vad_config.max_recording_seconds):
                break
        self.listening = False
        chunks = self.buffer.drain()
        audio = b"".join(chunk.data for chunk in chunks)
        duration_ms = max(
            int((time.monotonic() - started) * 1000),
            len(chunks) * self.audio_config.chunk_duration_ms,
        )
        if not speech_started:
            self.state = "timeout"
            self._event("audio_state", {"state": self.state})
            return {"ok": False, "status": "timeout", "audio": b""}
        if duration_ms < max(self.audio_config.min_recording_ms, self.vad_config.min_recording_ms):
            self.state = "ready"
            return {"ok": False, "status": "too_short", "audio": b""}
        self.state = "processing"
        self._event("audio_state", {"state": self.state})
        return {"ok": True, "status": "recorded", "audio": audio}

    async def submit_voice_answer(self, session_id: str, transcript: str) -> dict[str, Any]:
        try:
            session = await self.teaching.submit_answer(session_id, StudentResponse(answer_text=transcript))
        except InvalidTransitionError:
            self.state = "ready"
            return {"ok": False, "status": "duplicate_or_invalid", "transcript": transcript}
        except TeachingError:
            self.state = "ready"
            return {"ok": False, "status": "teaching_session_missing", "transcript": transcript}
        tutor_text = ""
        for turn in reversed(session.turns):
            if turn.role == "tutor" and turn.text:
                tutor_text = turn.text
                break
        if tutor_text:
            await self.speak(tutor_text)
        return {"ok": True, "status": "submitted", "transcript": transcript, "session": session.model_dump(mode="json")}

    async def speak(self, text: str) -> dict[str, Any]:
        result = await self.tts.synthesize(text)
        self._event("tts_result", {"ok": result["ok"], "status": result["status"]})
        if not result["ok"]:
            self.state = "ready"
            return {"ok": False, "status": result["status"]}
        item_id = str(uuid4())
        try:
            self._speech_queue.put_nowait((item_id, text, result["audio"]))
        except asyncio.QueueFull:
            return {"ok": False, "status": "speech_queue_full"}
        return {"ok": True, "status": "queued", "utterance_id": item_id}

    async def _speech_worker(self) -> None:
        while True:
            item_id, text, audio = await self._speech_queue.get()
            self.speaking = True
            self.state = "speaking"
            self._event("speech_started", {"utterance_id": item_id})
            try:
                await self.speaker.speak(text, audio)
                self._event("speech_completed", {"utterance_id": item_id})
            finally:
                self.speaking = False
                if self.state == "speaking":
                    self.state = "ready"
                self._speech_queue.task_done()

    def _event(self, event_type: str, payload: dict[str, Any]) -> None:
        safe = {"type": event_type, "timestamp_utc": utc_now(), **payload}
        self.events.append(safe)

    def event_snapshot(self) -> list[dict[str, Any]]:
        return list(self.events)


def build_microphone(config: AudioConfig) -> MockMicrophoneDriver:
    if config.input_provider == "local":
        return LocalMicrophoneDriver(config)
    return MockMicrophoneDriver(config)


def build_speaker(config: AudioConfig) -> MockSpeakerDriver:
    if config.output_provider in {"local", "piper"}:
        return LocalSpeakerDriver()
    return MockSpeakerDriver()


def build_wake_word(config: WakeWordConfig) -> MockWakeWordDetector:
    if config.provider == "openwakeword":
        return OpenWakeWordDetector(config)
    return MockWakeWordDetector(config)


def build_stt(config: STTRuntimeConfig) -> MockSTTAdapter:
    if config.provider == "legacy":
        return LegacySTTAdapter(config)
    if config.provider == "whisper_cpp":
        return WhisperCppSTTAdapter(config)
    return MockSTTAdapter(config)


def build_tts(config: TTSRuntimeConfig) -> MockTTSAdapter:
    if config.provider == "legacy":
        return LegacyTTSAdapter(config)
    if config.provider == "piper":
        return PiperTTSAdapter(config)
    return MockTTSAdapter(config)


def resolve_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def build_silent_wav(duration_ms: int, sample_rate: int = 16000) -> bytes:
    frames = int(sample_rate * max(1, duration_ms) / 1000)
    pcm = b"\x00\x00" * frames
    import tempfile

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        temp_path = Path(tmp.name)
    try:
        with wave.open(str(temp_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm)
        return temp_path.read_bytes()
    finally:
        temp_path.unlink(missing_ok=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
