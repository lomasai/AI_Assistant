"""Mock runtime providers and hardware drivers for development and tests."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, AsyncIterator

from server.config import CameraConfig, ServoConfig
from server.interfaces import AudioChunk, Frame, MotionCommand, MotionResult, SensorSnapshot


class MockLLMProvider:
    name = "mock"

    async def generate(self, prompt: str, system_prompt: str | None = None, **_: Any) -> str:
        _ = system_prompt
        clean = " ".join(prompt.split()).strip()
        if not clean:
            clean = "I am ready."
        lowered = clean.lower()
        if "malformed_structured_output" in lowered:
            return "not-json"
        if "provider_failure" in lowered:
            raise RuntimeError("mock provider failure")
        if "teaching_structured_output" in lowered:
            return (
                '{"speech_text":"Let us learn this step by step.",'
                '"screen_title":"Mock Lesson",'
                '"screen_points":["Objective first","Example next","Then a quick check"],'
                '"expected_response_type":"spoken_or_text",'
                '"evaluation_criteria":["uses the key idea","answers clearly"],'
                '"suggested_next_state":"asking_question"}'
            )
        return f'{{"mode":"response","response":"Mock response: {clean[:120]}","action":null}}'


class MockCameraDriver:
    def __init__(self, config: CameraConfig | None = None) -> None:
        self.config = config or CameraConfig(provider="mock")
        self.started = False

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.started = False

    async def frames(self) -> AsyncIterator[Frame]:
        while self.started:
            yield Frame(
                data=None,
                width=self.config.width,
                height=self.config.height,
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                source="mock",
            )
            await asyncio.sleep(1 / max(1, self.config.preview_fps))


class BrowserCameraDriver(MockCameraDriver):
    """Placeholder registry entry for the existing getUserMedia workflow."""

    async def frames(self) -> AsyncIterator[Frame]:
        if False:
            yield Frame(data=None, width=self.config.width, height=self.config.height, timestamp_utc="", source="browser")
        return


class MockAudioInputDriver:
    async def listen(self) -> AsyncIterator[AudioChunk]:
        yield AudioChunk(
            data=b"",
            sample_rate=16000,
            channels=1,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
        )


class MockAudioOutputDriver:
    def __init__(self) -> None:
        self.spoken: list[str] = []

    async def speak(self, text: str) -> None:
        self.spoken.append(text)


class MockWakeWordEngine:
    async def wait_for_wake_word(self) -> bool:
        return True


class MockFaceDetector:
    async def detect(self, frame: Frame) -> list[dict[str, Any]]:
        _ = frame
        return []


class MockFaceRecognizer:
    async def recognize(self, frame: Frame, faces: list[dict[str, Any]]) -> list[dict[str, Any]]:
        _ = frame
        return [{"label": "Guest", "confidence": 0.0, "face": face} for face in faces]


class MockSensorProvider:
    async def read(self) -> SensorSnapshot:
        return SensorSnapshot(
            values={"temperature_c": 24.5, "humidity_percent": 48.0, "motion_detected": False},
            source="mock",
        )


class MockMotionController:
    def __init__(self, config: ServoConfig | None = None) -> None:
        self.config = config or ServoConfig()
        self.pan_deg = 0.0
        self.tilt_deg = 0.0

    async def move_camera(self, command: MotionCommand) -> MotionResult:
        self.pan_deg = min(max(command.pan_deg, self.config.pan_min_deg), self.config.pan_max_deg)
        self.tilt_deg = min(max(command.tilt_deg, self.config.tilt_min_deg), self.config.tilt_max_deg)
        return MotionResult(ok=True, pan_deg=self.pan_deg, tilt_deg=self.tilt_deg, message="mock_move")

    async def stop(self) -> MotionResult:
        return MotionResult(ok=True, pan_deg=self.pan_deg, tilt_deg=self.tilt_deg, message="mock_stop")


class InMemoryStudentRepository:
    def __init__(self) -> None:
        self.students: dict[str, dict[str, Any]] = {}

    async def get_student(self, student_id: str) -> dict[str, Any] | None:
        return self.students.get(student_id)


class InMemorySessionRepository:
    def __init__(self) -> None:
        self.sessions: list[dict[str, Any]] = []
        self.session_map: dict[str, dict[str, Any]] = {}

    async def create_session(self, student_id: str | None, topic: str) -> dict[str, Any]:
        session = {
            "id": str(len(self.sessions) + 1),
            "student_id": student_id,
            "topic": topic,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        self.sessions.append(session)
        self.session_map[session["id"]] = dict(session)
        return session

    async def save_session(self, session: dict[str, Any]) -> None:
        session_id = str(session.get("id", ""))
        if not session_id:
            return
        self.session_map[session_id] = dict(session)
        for index, existing in enumerate(self.sessions):
            if str(existing.get("id")) == session_id:
                self.sessions[index] = dict(session)
                break
        else:
            self.sessions.append(dict(session))

    async def get_session(self, session_id: str) -> dict[str, Any] | None:
        session = self.session_map.get(session_id)
        return dict(session) if session else None
