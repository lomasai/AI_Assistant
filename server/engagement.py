"""Observable engagement signal runtime and non-punitive intervention policy."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from server.config import EngagementConfig, PROJECT_ROOT
from server.teaching import TeachingError, TeachingOrchestrator, TeachingSession


EngagementState = Literal[
    "disabled",
    "normal",
    "possible_absence",
    "gentle_prompt",
    "question_repeat",
    "short_recap",
    "teacher_assistance_suggested",
]
HeadOrientation = Literal["center", "left", "right", "up", "down", "unknown"]
InterventionChoice = Literal["continue", "repeat", "pause", "use_text"]


class EngagementError(Exception):
    """Raised when the engagement pipeline cannot run as configured."""


class ObservableSignal(BaseModel):
    """Coarse observable signal payload.

    This model intentionally excludes images, embeddings, confidence values,
    head angles and inferred psychological labels.
    """

    model_config = ConfigDict(extra="forbid")

    timestamp_utc: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    face_present: bool = True
    multiple_faces: bool = False
    outside_frame: bool = False
    head_orientation: HeadOrientation = "unknown"
    response_delay_seconds: float | None = Field(default=None, ge=0.0)
    unclear_answer_count: int = Field(default=0, ge=0)
    inactivity_count: int = Field(default=0, ge=0)


@dataclass(slots=True)
class InterventionEvent:
    event_id: str
    session_id: str
    state: EngagementState
    message: str
    reason: str
    timestamp_utc: str

    def safe_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "session_id": self.session_id,
            "state": self.state,
            "message": self.message,
            "reason": self.reason,
            "timestamp_utc": self.timestamp_utc,
        }


class EngagementSignalProvider(Protocol):
    provider_name: str

    async def initialize(self) -> None:
        ...

    async def close(self) -> None:
        ...

    def health(self) -> dict[str, Any]:
        ...


class MockEngagementSignalProvider:
    provider_name = "mock"

    def __init__(self) -> None:
        self.ready = False

    async def initialize(self) -> None:
        self.ready = True

    async def close(self) -> None:
        self.ready = False

    def health(self) -> dict[str, Any]:
        return {"provider": self.provider_name, "ready": self.ready}


class LocalEngagementSignalProvider:
    provider_name = "local"

    def __init__(self, config: EngagementConfig) -> None:
        self.config = config
        self.ready = False

    async def initialize(self) -> None:
        model_path = _resolve_model_path(self.config.model_path)
        if not model_path.exists():
            raise EngagementError("Configured engagement model is unavailable.")
        try:
            __import__("cv2")
        except Exception as exc:  # noqa: BLE001
            raise EngagementError("OpenCV is required for the local engagement provider.") from exc
        self.ready = True

    async def close(self) -> None:
        self.ready = False

    def health(self) -> dict[str, Any]:
        return {"provider": self.provider_name, "ready": self.ready}


@dataclass
class RollingSignalWindow:
    window_seconds: float
    signals: deque[ObservableSignal] = field(default_factory=deque)

    def add(self, signal: ObservableSignal) -> None:
        self.signals.append(signal)
        self._trim(signal.timestamp_utc)

    def clear(self) -> None:
        self.signals.clear()

    def abnormal_ratio(self) -> float:
        if not self.signals:
            return 0.0
        abnormal = 0
        for signal in self.signals:
            if (
                not signal.face_present
                or signal.multiple_faces
                or signal.outside_frame
                or signal.head_orientation in {"left", "right", "up", "down"}
            ):
                abnormal += 1
        return abnormal / len(self.signals)

    def absence_seconds(self) -> float:
        absent = [item for item in self.signals if not item.face_present or item.outside_frame]
        if len(absent) < 2:
            return 0.0
        return max(0.0, _parse_time(absent[-1].timestamp_utc) - _parse_time(absent[0].timestamp_utc))

    def multiple_face_seen(self) -> bool:
        return any(signal.multiple_faces for signal in self.signals)

    def latest(self) -> ObservableSignal | None:
        return self.signals[-1] if self.signals else None

    def _trim(self, now: str) -> None:
        cutoff = _parse_time(now) - self.window_seconds
        while self.signals and _parse_time(self.signals[0].timestamp_utc) < cutoff:
            self.signals.popleft()


@dataclass
class SessionEngagementState:
    enabled: bool = True
    state: EngagementState = "normal"
    message: str = ""
    reason: str = "normal"
    interventions_used: int = 0
    last_intervention_at: float = 0.0
    window: RollingSignalWindow | None = None
    history: deque[InterventionEvent] = field(default_factory=deque)


class EngagementInterventionPolicy:
    """Conservative, deterministic policy for supportive lesson interventions."""

    def __init__(self, config: EngagementConfig) -> None:
        self.config = config

    def evaluate(
        self,
        state: SessionEngagementState,
        signal: ObservableSignal,
        *,
        session: TeachingSession,
        audio_speaking: bool,
    ) -> tuple[EngagementState, str, str] | None:
        if not state.enabled or session.state in {"paused", "session_complete", "error"}:
            return None
        if audio_speaking:
            return None
        if state.interventions_used >= self.config.max_interventions_per_lesson:
            return None
        if len(state.window.signals if state.window else []) < self.config.minimum_samples:
            return None
        now = _parse_time(signal.timestamp_utc)
        if state.last_intervention_at and now - state.last_intervention_at < self.config.intervention_cooldown_seconds:
            return None

        if state.window and state.window.multiple_face_seen():
            return ("teacher_assistance_suggested", "A teacher can help if you want.", "multiple_faces")
        if state.window and state.window.absence_seconds() >= self.config.absence_duration_seconds:
            return ("possible_absence", "I will wait until you are ready.", "prolonged_absence")
        if signal.response_delay_seconds is not None and signal.response_delay_seconds >= self.config.response_inactivity_seconds:
            return ("gentle_prompt", "Ready when you are.", "response_delay")
        if signal.unclear_answer_count >= 2:
            return ("question_repeat", "Would you like me to repeat that?", "unclear_answers")
        if signal.inactivity_count >= 2:
            return ("short_recap", "Let us try a simpler example.", "inactivity")
        if state.window and state.window.abnormal_ratio() >= self.config.abnormal_sample_ratio:
            return ("gentle_prompt", "Ready when you are.", "observable_change")
        return None


class EngagementRuntime:
    """Owns engagement signal state and emits safe intervention events."""

    def __init__(
        self,
        config: EngagementConfig,
        teaching: TeachingOrchestrator,
        audio_status: Any,
        provider: EngagementSignalProvider | None = None,
    ) -> None:
        self.config = config
        self.teaching = teaching
        self._audio_status = audio_status
        self.provider = provider or build_engagement_provider(config)
        self.policy = EngagementInterventionPolicy(config)
        self._sessions: dict[str, SessionEngagementState] = {}
        self._events: deque[InterventionEvent] = deque(maxlen=config.event_retention)
        self._lock = asyncio.Lock()
        self._running = False

    async def start(self) -> None:
        if self.config.provider == "disabled" or not self.config.enabled:
            self._running = False
            return
        await self.provider.initialize()
        self._running = True

    async def stop(self) -> None:
        await self.provider.close()
        self._running = False

    def health(self) -> dict[str, Any]:
        return {
            "enabled": self.config.enabled,
            "running": self._running,
            "provider": self.config.provider,
            "analysis_fps": self.config.analysis_fps,
            "retains_raw_frames": False,
            "provider_health": self.provider.health(),
        }

    async def set_session_enabled(self, session_id: str, enabled: bool) -> dict[str, Any]:
        async with self._lock:
            state = self._state_for(session_id)
            state.enabled = enabled and self.config.enabled
            if not state.enabled:
                state.state = "disabled"
                state.message = ""
                state.reason = "disabled"
                if state.window:
                    state.window.clear()
            elif state.state == "disabled":
                state.state = "normal"
                state.reason = "normal"
        return await self.current_state(session_id)

    async def ingest_signal(self, session_id: str, signal: ObservableSignal) -> dict[str, Any]:
        async with self._lock:
            session_state = self._state_for(session_id)
            if not self.config.enabled or not session_state.enabled:
                session_state.state = "disabled"
                return _current_state_dict(session_id, session_state)
            session_state.window.add(signal)

            try:
                session = await self.teaching.get_session(session_id)
            except TeachingError:
                return _current_state_dict(session_id, session_state)

            if _participation_resumed(signal):
                session_state.state = "normal"
                session_state.message = ""
                session_state.reason = "normal"

            decision = self.policy.evaluate(
                session_state,
                signal,
                session=session,
                audio_speaking=bool(self._audio_status().speaking),
            )
            if decision is not None:
                next_state, message, reason = decision
                session_state.state = next_state
                session_state.message = message
                session_state.reason = reason
                session_state.interventions_used += 1
                session_state.last_intervention_at = _parse_time(signal.timestamp_utc)
                event = InterventionEvent(
                    event_id=str(uuid4()),
                    session_id=session_id,
                    state=next_state,
                    message=message,
                    reason=reason,
                    timestamp_utc=signal.timestamp_utc,
                )
                session_state.history.append(event)
                while len(session_state.history) > self.config.event_retention:
                    session_state.history.popleft()
                self._events.append(event)
                await self.teaching.record_intervention(session_id, next_state, message)
            return _current_state_dict(session_id, session_state)

    async def current_state(self, session_id: str) -> dict[str, Any]:
        async with self._lock:
            return _current_state_dict(session_id, self._state_for(session_id))

    async def history(self, session_id: str) -> dict[str, Any]:
        async with self._lock:
            state = self._state_for(session_id)
            return {"session_id": session_id, "events": [event.safe_dict() for event in state.history]}

    async def handle_choice(self, session_id: str, choice: InterventionChoice) -> dict[str, Any]:
        if choice == "pause":
            await self.teaching.pause(session_id)
        async with self._lock:
            state = self._state_for(session_id)
            if choice in {"continue", "use_text"}:
                state.state = "normal"
                state.message = ""
                state.reason = "normal"
                if state.window:
                    state.window.clear()
            elif choice == "repeat":
                state.state = "question_repeat"
                state.message = "Would you like me to repeat that?"
                state.reason = "student_choice"
            return _current_state_dict(session_id, state)

    def events(self) -> list[dict[str, Any]]:
        return [event.safe_dict() for event in self._events]

    def _state_for(self, session_id: str) -> SessionEngagementState:
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionEngagementState(
                enabled=self.config.enabled,
                state="normal" if self.config.enabled else "disabled",
                window=RollingSignalWindow(self.config.rolling_window_seconds),
            )
        return self._sessions[session_id]


def build_engagement_provider(config: EngagementConfig) -> EngagementSignalProvider:
    if config.provider == "local":
        return LocalEngagementSignalProvider(config)
    return MockEngagementSignalProvider()


def _current_state_dict(session_id: str, state: SessionEngagementState) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "enabled": state.enabled,
        "state": state.state,
        "message": state.message,
        "reason": state.reason,
        "interventions_used": state.interventions_used,
    }


def _participation_resumed(signal: ObservableSignal) -> bool:
    return (
        signal.face_present
        and not signal.multiple_faces
        and not signal.outside_frame
        and signal.head_orientation in {"center", "unknown"}
        and (signal.response_delay_seconds or 0.0) == 0.0
        and signal.unclear_answer_count == 0
        and signal.inactivity_count == 0
    )


def _parse_time(value: str) -> float:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return datetime.now(timezone.utc).timestamp()


def _resolve_model_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path
