import unittest

from server.config import RuntimeConfig
from server.interfaces import MotionCommand
from server.llm.providers import OpenAICompatibleProvider
from server.mock_drivers import MockLLMProvider
from server.runtime import build_application_runtime


class TestApplicationRuntime(unittest.IsolatedAsyncioTestCase):
    async def test_registry_selects_mock_and_openai_compatible_profiles(self) -> None:
        config = RuntimeConfig.model_validate(
            {
                "llm": {
                    "active_provider": "mock",
                    "profiles": {
                        "mock": {"provider": "mock", "model": "mock"},
                        "groq": {
                            "provider": "openai_compatible",
                            "base_url": "https://api.groq.com/openai/v1",
                            "model": "llama-test",
                            "api_key_env": "GROQ_API_KEY",
                        },
                    },
                }
            }
        )

        runtime = build_application_runtime(config)

        self.assertIsInstance(runtime.llms.get("mock"), MockLLMProvider)
        self.assertIsInstance(runtime.llms.get("groq"), OpenAICompatibleProvider)
        self.assertIn("groq", runtime.health()["registered_llm_providers"])

    async def test_mock_drivers_are_available(self) -> None:
        config = RuntimeConfig.model_validate(
            {"llm": {"active_provider": "mock", "profiles": {"mock": {"provider": "mock", "model": "mock"}}}}
        )
        runtime = build_application_runtime(config)

        await runtime.drivers.camera.start()
        audio_chunks = []
        async for chunk in runtime.drivers.audio_input.listen():
            audio_chunks.append(chunk)
            break
        motion = await runtime.drivers.motion.move_camera(MotionCommand(pan_deg=100, tilt_deg=-100))
        sensor = await runtime.drivers.sensors.read()
        faces = await runtime.drivers.face_detector.detect(
            frame=type("FrameStub", (), {"data": None, "width": 640, "height": 480})()
        )
        recognized = await runtime.drivers.face_recognizer.recognize(
            frame=type("FrameStub", (), {"data": None, "width": 640, "height": 480})(),
            faces=[{"x": 1, "y": 2, "width": 3, "height": 4}],
        )
        student = await runtime.drivers.students.get_student("missing")
        session = await runtime.drivers.sessions.create_session(student_id=None, topic="fractions")
        await runtime.drivers.audio_output.speak("hello")
        wake = await runtime.drivers.wake_word.wait_for_wake_word()
        await runtime.drivers.camera.stop()

        self.assertEqual(len(audio_chunks), 1)
        self.assertTrue(motion.ok)
        self.assertEqual(motion.pan_deg, 45.0)
        self.assertEqual(motion.tilt_deg, -20.0)
        self.assertEqual(sensor.source, "mock")
        self.assertEqual(faces, [])
        self.assertEqual(recognized[0]["label"], "Guest")
        self.assertIsNone(student)
        self.assertEqual(session["topic"], "fractions")
        self.assertTrue(wake)


if __name__ == "__main__":
    unittest.main()
