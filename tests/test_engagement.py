import asyncio
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi.testclient import TestClient
from pydantic import ValidationError

from api.main import create_app
from server.config import EngagementConfig, RuntimeConfig
from server.engagement import EngagementError, EngagementRuntime, ObservableSignal, RollingSignalWindow
from server.runtime import build_application_runtime
from server.teaching import LessonConfig


def ts(offset_seconds: float) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=offset_seconds)).isoformat()


def config(db_path: Path, extra: dict | None = None) -> RuntimeConfig:
    raw = {
        "llm": {"active_provider": "mock", "profiles": {"mock": {"provider": "mock", "model": "mock"}}},
        "camera": {"provider": "mock", "width": 160, "height": 120},
        "audio": {"input_provider": "mock", "output_provider": "mock", "min_recording_ms": 0},
        "wake_word": {"provider": "mock", "cooldown_seconds": 0.1, "activation_timeout_seconds": 0.1},
        "vad": {"provider": "mock", "min_recording_ms": 0, "silence_timeout_ms": 1},
        "stt": {"provider": "mock"},
        "tts": {"provider": "mock"},
        "database": {"sqlite_path": str(db_path)},
        "engagement": {
            "enabled": True,
            "provider": "mock",
            "analysis_fps": 2,
            "rolling_window_seconds": 10,
            "absence_duration_seconds": 2,
            "response_inactivity_seconds": 5,
            "intervention_cooldown_seconds": 0,
            "max_interventions_per_lesson": 3,
            "minimum_samples": 3,
        },
        "feature_flags": {
            "student_ui": True,
            "text_input": True,
            "backend_camera_stream": True,
            "engagement_analysis": True,
        },
    }
    if extra:
        for key, value in extra.items():
            if isinstance(value, dict) and isinstance(raw.get(key), dict):
                raw[key].update(value)
            else:
                raw[key] = value
    return RuntimeConfig.model_validate(raw)


def lesson() -> LessonConfig:
    return LessonConfig(
        student_display_name="Asha",
        grade_level="grade_6",
        topic="Fractions",
        language="en",
        objective="Understand what a fraction means.",
    )


class TestEngagementRuntime(unittest.IsolatedAsyncioTestCase):
    async def make_runtime(self) -> tuple[tempfile.TemporaryDirectory, object]:
        tmp = tempfile.TemporaryDirectory()
        runtime = build_application_runtime(config(Path(tmp.name) / "engagement.db"))
        await runtime.student_store.initialize()
        await runtime.engagement.start()
        return tmp, runtime

    async def test_mock_provider_lifecycle_and_health(self) -> None:
        tmp, runtime = await self.make_runtime()
        try:
            health = runtime.engagement.health()
            self.assertTrue(health["enabled"])
            self.assertTrue(health["running"])
            self.assertFalse(health["retains_raw_frames"])
            self.assertNotIn("model_path", str(health))
        finally:
            await runtime.engagement.stop()
            tmp.cleanup()

    async def test_smoothing_prevents_single_frame_intervention(self) -> None:
        tmp, runtime = await self.make_runtime()
        try:
            session = await runtime.teaching.create_session(lesson())
            await runtime.teaching.start(session.id)
            state = await runtime.engagement.ingest_signal(
                session.id,
                ObservableSignal(timestamp_utc=ts(0), face_present=False),
            )
        finally:
            await runtime.engagement.stop()
            tmp.cleanup()

        self.assertEqual(state["state"], "normal")
        self.assertEqual(state["interventions_used"], 0)

    async def test_absence_and_multiple_face_generate_neutral_interventions(self) -> None:
        tmp, runtime = await self.make_runtime()
        try:
            session = await runtime.teaching.create_session(lesson())
            await runtime.teaching.start(session.id)
            for index in range(3):
                absence = await runtime.engagement.ingest_signal(
                    session.id,
                    ObservableSignal(timestamp_utc=ts(index), face_present=False, outside_frame=True),
                )
            self.assertEqual(absence["state"], "possible_absence")
            self.assertEqual(absence["message"], "I will wait until you are ready.")

            for index in range(3, 6):
                multiple = await runtime.engagement.ingest_signal(
                    session.id,
                    ObservableSignal(timestamp_utc=ts(index + 20), face_present=True, multiple_faces=True),
                )
            self.assertEqual(multiple["state"], "teacher_assistance_suggested")
            self.assertNotIn("distracted", str(multiple).lower())
        finally:
            await runtime.engagement.stop()
            tmp.cleanup()

    async def test_cooldown_max_and_participation_recovery(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        runtime = build_application_runtime(
            config(
                Path(tmp.name) / "engagement.db",
                {"engagement": {"intervention_cooldown_seconds": 30, "max_interventions_per_lesson": 1}},
            )
        )
        await runtime.student_store.initialize()
        await runtime.engagement.start()
        try:
            session = await runtime.teaching.create_session(lesson())
            await runtime.teaching.start(session.id)
            for index in range(3):
                first = await runtime.engagement.ingest_signal(
                    session.id,
                    ObservableSignal(timestamp_utc=ts(index), response_delay_seconds=10),
                )
            self.assertEqual(first["state"], "gentle_prompt")
            resumed = await runtime.engagement.ingest_signal(
                session.id,
                ObservableSignal(timestamp_utc=ts(40), face_present=True, head_orientation="center"),
            )
            self.assertEqual(resumed["state"], "normal")
            for index in range(41, 44):
                capped = await runtime.engagement.ingest_signal(
                    session.id,
                    ObservableSignal(timestamp_utc=ts(index), unclear_answer_count=3),
                )
            self.assertEqual(capped["interventions_used"], 1)
        finally:
            await runtime.engagement.stop()
            tmp.cleanup()

    async def test_no_prompt_while_audio_speaking_or_session_paused(self) -> None:
        tmp, runtime = await self.make_runtime()
        try:
            session = await runtime.teaching.create_session(lesson())
            await runtime.teaching.start(session.id)
            runtime.audio.speaking = True
            for index in range(3):
                state = await runtime.engagement.ingest_signal(
                    session.id,
                    ObservableSignal(timestamp_utc=ts(index), response_delay_seconds=10),
                )
            self.assertEqual(state["state"], "normal")
            runtime.audio.speaking = False
            await runtime.teaching.pause(session.id)
            for index in range(4, 7):
                paused = await runtime.engagement.ingest_signal(
                    session.id,
                    ObservableSignal(timestamp_utc=ts(index), unclear_answer_count=3),
                )
            self.assertEqual(paused["state"], "normal")
        finally:
            await runtime.engagement.stop()
            tmp.cleanup()

    async def test_intervention_is_persisted_with_teaching_session(self) -> None:
        tmp, runtime = await self.make_runtime()
        try:
            session = await runtime.teaching.create_session(lesson())
            await runtime.teaching.start(session.id)
            for index in range(3):
                await runtime.engagement.ingest_signal(
                    session.id,
                    ObservableSignal(timestamp_utc=ts(index), unclear_answer_count=2),
                )
            reloaded = await runtime.teaching.get_session(session.id)
            system_text = " ".join(turn.text for turn in reloaded.turns if turn.role == "system")
        finally:
            await runtime.engagement.stop()
            tmp.cleanup()

        self.assertIn("Engagement support", system_text)
        self.assertIn("Would you like me to repeat that?", system_text)


class TestEngagementApi(unittest.TestCase):
    def test_api_state_history_privacy_and_regression(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = build_application_runtime(config(Path(tmp) / "engagement.db"))
            app = create_app(runtime)
            with TestClient(app) as client:
                created = client.post(
                    "/api/v1/teaching/sessions",
                    json=lesson().model_dump(mode="json"),
                )
                self.assertEqual(created.status_code, 200)
                session_id = created.json()["session"]["id"]
                self.assertEqual(client.post(f"/api/v1/teaching/sessions/{session_id}/start").status_code, 200)
                for index in range(3):
                    response = client.post(
                        f"/api/v1/engagement/sessions/{session_id}/signals",
                        json={"timestamp_utc": ts(index), "unclear_answer_count": 2},
                    )
                self.assertEqual(response.status_code, 200)
                body = response.json()
                self.assertEqual(body["state"], "question_repeat")
                forbidden = str(body).lower()
                for word in ("confidence", "angle", "embedding", "model_path", "distracted", "lazy", "adhd", "emotion"):
                    self.assertNotIn(word, forbidden)

                history = client.get(f"/api/v1/engagement/sessions/{session_id}/history")
                self.assertEqual(history.status_code, 200)
                self.assertIn("events", history.json())
                self.assertEqual(client.post("/chat", json={"user_text": "hello", "store_log": False}).status_code, 200)
                self.assertEqual(client.get("/camera/status").status_code, 200)

    def test_enable_choice_and_health_endpoints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = build_application_runtime(config(Path(tmp) / "engagement.db"))
            app = create_app(runtime)
            with TestClient(app) as client:
                created = client.post("/api/v1/teaching/sessions", json=lesson().model_dump(mode="json")).json()
                session_id = created["session"]["id"]
                self.assertEqual(client.get("/api/v1/engagement/health").status_code, 200)
                disabled = client.post(f"/api/v1/engagement/sessions/{session_id}/enable", json={"enabled": False})
                self.assertEqual(disabled.json()["state"], "disabled")
                enabled = client.post(f"/api/v1/engagement/sessions/{session_id}/enable", json={"enabled": True})
                self.assertEqual(enabled.json()["state"], "normal")
                choice = client.post(f"/api/v1/engagement/sessions/{session_id}/choice", json={"choice": "use_text"})
                self.assertEqual(choice.status_code, 200)


class TestEngagementConfig(unittest.TestCase):
    def test_invalid_local_provider_and_raw_frame_retention_fail(self) -> None:
        with self.assertRaises(ValidationError):
            EngagementConfig(provider="local", model_path="")
        with self.assertRaises(ValidationError):
            EngagementConfig(retain_raw_frames=True)

    def test_local_provider_missing_model_fails_clearly(self) -> None:
        runtime = EngagementRuntime(
            EngagementConfig(enabled=True, provider="local", model_path="models/missing.onnx"),
            teaching=None,
            audio_status=lambda: object(),
        )
        with self.assertRaises(EngagementError):
            asyncio.run(runtime.start())

    def test_rolling_window_drops_stale_samples(self) -> None:
        window = RollingSignalWindow(window_seconds=2)
        window.add(ObservableSignal(timestamp_utc=ts(0), face_present=False))
        window.add(ObservableSignal(timestamp_utc=ts(1), face_present=False))
        window.add(ObservableSignal(timestamp_utc=ts(5), face_present=True))

        self.assertEqual(len(window.signals), 1)
        self.assertEqual(window.abnormal_ratio(), 0.0)


if __name__ == "__main__":
    unittest.main()
