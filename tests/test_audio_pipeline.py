import asyncio
import base64
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from api.main import create_app
from server.audio_pipeline import AudioRuntime, BoundedAudioBuffer, MockMicrophoneDriver
from server.config import RuntimeConfig
from server.interfaces import AudioChunk
from server.runtime import build_application_runtime
from server.teaching import LessonConfig


def audio_config(db_path: Path, extra: dict | None = None) -> RuntimeConfig:
    raw = {
        "llm": {"active_provider": "mock", "profiles": {"mock": {"provider": "mock", "model": "mock"}}},
        "camera": {"provider": "mock", "width": 160, "height": 120},
        "audio": {"input_provider": "mock", "output_provider": "mock", "min_recording_ms": 0},
        "wake_word": {"provider": "mock", "cooldown_seconds": 1, "activation_timeout_seconds": 1},
        "vad": {"provider": "mock", "min_recording_ms": 0, "silence_timeout_ms": 1},
        "stt": {"provider": "mock", "mock_transcript": "A fraction is part of a whole."},
        "tts": {"provider": "mock"},
        "database": {"sqlite_path": str(db_path)},
        "feature_flags": {
            "push_to_talk": True,
            "voice_turns": True,
            "wake_word": True,
            "stt_input": True,
            "tts_output": True,
            "student_ui": True,
            "backend_camera_stream": True,
            "text_input": True,
        },
    }
    if extra:
        for key, value in extra.items():
            if isinstance(value, dict) and isinstance(raw.get(key), dict):
                raw[key].update(value)
            else:
                raw[key] = value
    return RuntimeConfig.model_validate(raw)


class TestAudioPipeline(unittest.IsolatedAsyncioTestCase):
    async def make_runtime(self) -> tuple[tempfile.TemporaryDirectory, AudioRuntime]:
        tmp = tempfile.TemporaryDirectory()
        app_runtime = build_application_runtime(audio_config(Path(tmp.name) / "audio.db"))
        await app_runtime.student_store.initialize()
        await app_runtime.audio.start()
        return tmp, app_runtime.audio

    async def test_mock_microphone_and_speaker_lifecycle(self) -> None:
        tmp, audio = await self.make_runtime()
        try:
            self.assertEqual(audio.status().state, "ready")
            await audio.speak("hello")
            await asyncio.sleep(0.01)
            self.assertIn("hello", audio.speaker.spoken)
        finally:
            await audio.stop()
            tmp.cleanup()
        self.assertEqual(audio.status().state, "stopped")

    async def test_audio_buffer_limits_and_stale_drop(self) -> None:
        buffer = BoundedAudioBuffer(max_chunks=2)
        for index in range(3):
            buffer.append(AudioChunk(data=bytes([index]), sample_rate=16000, channels=1, timestamp_utc="now"))
        drained = buffer.drain()

        self.assertEqual(buffer.dropped_chunks, 1)
        self.assertEqual([chunk.data for chunk in drained], [b"\x01", b"\x02"])

    async def test_wake_word_activation_cooldown_and_false_rejection(self) -> None:
        tmp, audio = await self.make_runtime()
        try:
            first = await audio.wait_for_wake_word()
            second = await audio.wait_for_wake_word()
            audio.wake_word.force_false = True
            await asyncio.sleep(1.01)
            false = await audio.wait_for_wake_word()
        finally:
            await audio.stop()
            tmp.cleanup()

        self.assertTrue(first["activated"])
        self.assertFalse(second["activated"])
        self.assertFalse(false["activated"])

    async def test_vad_speech_start_end_and_timeout(self) -> None:
        tmp, audio = await self.make_runtime()
        try:
            recorded = await audio.record_until_silence()
            audio.microphone = MockMicrophoneDriver(audio.audio_config, chunks=[b"\x00" * 64])
            await audio.microphone.start()
            timeout = await audio.record_until_silence()
        finally:
            await audio.stop()
            tmp.cleanup()

        self.assertTrue(recorded["ok"])
        self.assertEqual(timeout["status"], "timeout")

    async def test_successful_and_failed_stt_and_tts_queue_cancel_completion(self) -> None:
        tmp, audio = await self.make_runtime()
        try:
            ok = await audio.push_to_talk()
            audio.stt.fail_next = True
            failed_stt = await audio.push_to_talk()
            queued = await audio.speak("queued speech")
            await asyncio.sleep(0.01)
            audio.tts.fail_next = True
            failed_tts = await audio.speak("will fail")
            cancelled = await audio.cancel()
        finally:
            await audio.stop()
            tmp.cleanup()

        self.assertTrue(ok["ok"])
        self.assertEqual(ok["transcript"], "A fraction is part of a whole.")
        self.assertEqual(failed_stt["status"], "stt_failed")
        self.assertEqual(queued["status"], "queued")
        self.assertEqual(failed_tts["status"], "tts_failed")
        self.assertEqual(cancelled["state"], "ready")

    async def test_complete_voice_teaching_turn_and_duplicate_prevention(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        app_runtime = build_application_runtime(audio_config(Path(tmp.name) / "voice.db"))
        await app_runtime.student_store.initialize()
        await app_runtime.audio.start()
        try:
            lesson = LessonConfig(
                student_display_name="Asha",
                grade_level="6",
                topic="Fractions",
                language="en",
                objective="Understand fractions.",
            )
            created = await app_runtime.teaching.create_session(lesson)
            await app_runtime.teaching.start(created.id)
            result = await app_runtime.audio.push_to_talk(session_id=created.id)
            duplicate = await app_runtime.audio.submit_voice_answer(created.id, "same answer")
        finally:
            await app_runtime.audio.stop()
            tmp.cleanup()

        self.assertTrue(result["ok"])
        self.assertEqual(result["session"]["state"], "session_complete")
        self.assertEqual(duplicate["status"], "duplicate_or_invalid")

    async def test_pause_stop_cancel_active_audio(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        app_runtime = build_application_runtime(audio_config(Path(tmp.name) / "cancel.db"))
        await app_runtime.student_store.initialize()
        await app_runtime.audio.start()
        try:
            await app_runtime.audio.speak("active")
            await asyncio.sleep(0)
            cancelled = await app_runtime.audio.cancel()
        finally:
            await app_runtime.audio.stop()
            tmp.cleanup()

        self.assertFalse(cancelled["speaking"])

    async def test_provider_unavailable_behaviour(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        cfg = audio_config(
            Path(tmp.name) / "unavailable.db",
            {"audio": {"input_provider": "local", "output_provider": "mock"}},
        )
        app_runtime = build_application_runtime(cfg)
        await app_runtime.audio.start()
        try:
            status = app_runtime.audio.status()
        finally:
            await app_runtime.audio.stop()
            tmp.cleanup()

        self.assertEqual(status.state, "error")
        self.assertFalse(status.microphone_available)


class TestAudioApi(unittest.TestCase):
    def test_audio_api_voice_turn_shutdown_and_legacy_compatibility(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = build_application_runtime(audio_config(Path(tmp) / "api.db"))
            app = create_app(runtime)
            with TestClient(app) as client:
                lesson = client.post(
                    "/api/v1/teaching/sessions",
                    json={
                        "student_display_name": "Asha",
                        "grade_level": "6",
                        "topic": "Fractions",
                        "language": "en",
                        "objective": "Understand fractions.",
                    },
                )
                session_id = lesson.json()["session"]["id"]
                client.post(f"/api/v1/teaching/sessions/{session_id}/start")
                voice = client.post("/api/v1/audio/push-to-talk/start", json={"session_id": session_id})
                health = client.get("/api/v1/audio/health")
                state = client.get("/api/v1/audio/state")
                wake = client.post("/api/v1/audio/wake-word/activate")
                tts = client.post("/api/v1/audio/tts/start", json={"text": "hello"})
                cancel = client.post("/api/v1/audio/tts/cancel")
                legacy_stt = client.post(
                    "/stt",
                    json={"audio_base64": base64.b64encode(b"abc").decode("ascii"), "filename": "a.wav"},
                )
                legacy_tts = client.post("/tts", json={"text": "hello", "output_mode": "base64"})

            self.assertEqual(voice.status_code, 200)
            self.assertTrue(voice.json()["ok"])
            self.assertEqual(health.status_code, 200)
            self.assertEqual(state.status_code, 200)
            self.assertIn(wake.status_code, {200})
            self.assertEqual(tts.status_code, 200)
            self.assertEqual(cancel.status_code, 200)
            self.assertEqual(legacy_stt.status_code, 200)
            self.assertEqual(legacy_tts.status_code, 200)


if __name__ == "__main__":
    unittest.main()
