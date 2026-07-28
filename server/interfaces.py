"""Provider and hardware protocols used by the application runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Protocol, runtime_checkable


@dataclass(slots=True)
class Frame:
    data: Any
    width: int
    height: int
    timestamp_utc: str
    source: str = "mock"
    sequence: int = 0
    jpeg_bytes: bytes | None = None


@dataclass(slots=True)
class AudioChunk:
    data: bytes
    sample_rate: int
    channels: int
    timestamp_utc: str
    sequence: int = 0
    speech_hint: bool = False


@dataclass(slots=True)
class MotionCommand:
    pan_deg: float
    tilt_deg: float
    speed: float = 20.0


@dataclass(slots=True)
class MotionResult:
    ok: bool
    pan_deg: float
    tilt_deg: float
    message: str


@dataclass(slots=True)
class HardwareCommand:
    command_id: str
    protocol_version: str
    timestamp_utc: str
    action: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class HardwareAck:
    command_id: str
    ok: bool
    status: str
    timestamp_utc: str
    detail: str = ""


@dataclass(slots=True)
class SensorSnapshot:
    values: dict[str, Any]
    source: str
    status: str = "ok"


@dataclass(slots=True)
class LLMRequest:
    prompt: str
    system_prompt: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class LLMProvider(Protocol):
    name: str

    async def generate(self, prompt: str, system_prompt: str | None = None, **kwargs: Any) -> str:
        """Generate one assistant response."""


@runtime_checkable
class CameraDriver(Protocol):
    async def start(self) -> None:
        """Start camera capture."""

    async def stop(self) -> None:
        """Stop camera capture."""

    async def frames(self) -> AsyncIterator[Frame]:
        """Yield frames."""


@runtime_checkable
class AudioInputDriver(Protocol):
    async def listen(self) -> AsyncIterator[AudioChunk]:
        """Yield recorded audio chunks."""


@runtime_checkable
class AudioOutputDriver(Protocol):
    async def speak(self, text: str) -> None:
        """Speak or enqueue text output."""


@runtime_checkable
class WakeWordEngine(Protocol):
    async def wait_for_wake_word(self) -> bool:
        """Return true when wake word is detected."""


@runtime_checkable
class WakeWordDetector(Protocol):
    async def initialize(self) -> None:
        """Load wake-word resources."""

    async def detect(self, chunk: AudioChunk) -> bool:
        """Return true when a wake word is detected in the chunk."""

    def health(self) -> dict[str, Any]:
        """Return safe wake-word provider status."""


@runtime_checkable
class STTProvider(Protocol):
    async def transcribe_bytes(self, audio_bytes: bytes, filename: str = "audio.wav") -> str:
        """Transcribe audio bytes."""


@runtime_checkable
class TTSProvider(Protocol):
    async def synthesize_to_bytes(self, text: str) -> bytes:
        """Synthesize speech bytes."""


@runtime_checkable
class FaceDetector(Protocol):
    async def detect(self, frame: Frame) -> list[dict[str, Any]]:
        """Detect faces in a frame."""


@runtime_checkable
class FaceRecognizer(Protocol):
    async def recognize(self, frame: Frame, faces: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Recognize known faces."""


@runtime_checkable
class SensorProvider(Protocol):
    async def read(self) -> SensorSnapshot:
        """Read current sensor state."""


@runtime_checkable
class MotionController(Protocol):
    async def move_camera(self, command: MotionCommand) -> MotionResult:
        """Move camera toward a target pose."""

    async def stop(self) -> MotionResult:
        """Stop active movement."""


@runtime_checkable
class HardwareController(Protocol):
    async def start(self) -> None:
        """Initialize hardware command ownership."""

    async def stop(self) -> None:
        """Return hardware to a safe state and release resources."""

    async def submit(self, command: HardwareCommand) -> HardwareAck:
        """Submit one already-validated hardware command."""

    async def cancel(self) -> HardwareAck:
        """Cancel pending movement and enter a safe state."""

    async def emergency_stop(self) -> HardwareAck:
        """Immediately enter emergency-stop state."""

    def health(self) -> dict[str, Any]:
        """Return safe hardware health metadata."""


@runtime_checkable
class StudentRepository(Protocol):
    async def get_student(self, student_id: str) -> dict[str, Any] | None:
        """Fetch one student profile."""


@runtime_checkable
class SessionRepository(Protocol):
    async def create_session(self, student_id: str | None, topic: str) -> dict[str, Any]:
        """Create a teaching session record."""

    async def save_session(self, session: dict[str, Any]) -> None:
        """Persist a full teaching session snapshot."""

    async def get_session(self, session_id: str) -> dict[str, Any] | None:
        """Fetch a full teaching session snapshot."""
