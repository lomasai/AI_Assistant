import asyncio
import unittest

from fastapi.testclient import TestClient

from api.main import create_app
from server.config import RuntimeConfig, TeachingConfig
from server.mock_drivers import InMemorySessionRepository, MockLLMProvider
from server.teaching import (
    ALLOWED_TRANSITIONS,
    InvalidTransitionError,
    LessonConfig,
    StudentResponse,
    TeachingOrchestrator,
    TeachingSession,
)
from server.runtime import build_application_runtime


def lesson() -> LessonConfig:
    return LessonConfig(
        student_display_name="Asha",
        grade_level="grade_6",
        topic="Fractions",
        language="en",
        objective="Understand what a fraction means.",
    )


def runtime_config() -> RuntimeConfig:
    return RuntimeConfig.model_validate(
        {
            "llm": {"active_provider": "mock", "profiles": {"mock": {"provider": "mock", "model": "mock"}}},
            "camera": {"provider": "mock", "width": 160, "height": 120},
            "feature_flags": {"student_ui": True, "text_input": True, "backend_camera_stream": True},
        }
    )


class FailingProvider:
    name = "failing"

    async def generate(self, **kwargs):
        raise RuntimeError("provider failed")


class MalformedProvider:
    name = "malformed"

    async def generate(self, **kwargs):
        return "not-json"


class SlowProvider:
    name = "slow"

    async def generate(self, **kwargs):
        await asyncio.sleep(0.05)
        return "{}"


class TestTeachingOrchestrator(unittest.IsolatedAsyncioTestCase):
    async def make_orchestrator(self, provider=None, config=None):
        repo = InMemorySessionRepository()
        return TeachingOrchestrator(repo, provider or MockLLMProvider(), config or TeachingConfig())

    async def test_every_declared_valid_transition_is_allowed(self) -> None:
        orchestrator = await self.make_orchestrator()
        session = await orchestrator.create_session(lesson())

        for source, targets in ALLOWED_TRANSITIONS.items():
            for target in targets:
                candidate = TeachingSession.model_validate(session.model_dump(mode="json"))
                candidate.state = source
                orchestrator._transition(candidate, target)
                self.assertEqual(candidate.state, target)

    async def test_invalid_transition_is_controlled(self) -> None:
        orchestrator = await self.make_orchestrator()
        session = await orchestrator.create_session(lesson())

        with self.assertRaises(InvalidTransitionError):
            await orchestrator.submit_answer(session.id, StudentResponse(answer_text="yes because parts"))

    async def test_complete_successful_lesson(self) -> None:
        orchestrator = await self.make_orchestrator()
        session = await orchestrator.create_session(lesson())
        started = await orchestrator.start(session.id)

        self.assertEqual(started.state, "waiting_for_answer")

        completed = await orchestrator.submit_answer(session.id, StudentResponse(answer_text="yes because it is part of a whole"))

        self.assertEqual(completed.state, "session_complete")
        self.assertIsNotNone(completed.summary)
        self.assertEqual(completed.summary.evaluations[-1].label, "correct")

    async def test_incorrect_answer_followed_by_remediation(self) -> None:
        orchestrator = await self.make_orchestrator()
        session = await orchestrator.create_session(lesson())
        await orchestrator.start(session.id)

        result = await orchestrator.submit_answer(session.id, StudentResponse(answer_text="wrong"))

        self.assertEqual(result.state, "waiting_for_answer")
        self.assertEqual(result.progress.remediation_attempts, 1)
        self.assertIn("Try again", result.turns[-1].text)

    async def test_max_remediation_limit_completes_session(self) -> None:
        orchestrator = await self.make_orchestrator(config=TeachingConfig(max_remediation_attempts=0))
        session = await orchestrator.create_session(lesson())
        await orchestrator.start(session.id)

        result = await orchestrator.submit_answer(session.id, StudentResponse(answer_text="incorrect"))

        self.assertEqual(result.state, "session_complete")
        self.assertEqual(result.summary.evaluations[-1].label, "incorrect")

    async def test_pause_resume_and_stop(self) -> None:
        orchestrator = await self.make_orchestrator()
        session = await orchestrator.create_session(lesson())
        await orchestrator.start(session.id)

        paused = await orchestrator.pause(session.id)
        resumed = await orchestrator.resume(session.id)
        stopped = await orchestrator.stop(session.id)

        self.assertEqual(paused.state, "paused")
        self.assertEqual(resumed.state, "waiting_for_answer")
        self.assertEqual(stopped.state, "session_complete")

    async def test_duplicate_answer_submission_is_rejected_after_completion(self) -> None:
        orchestrator = await self.make_orchestrator()
        session = await orchestrator.create_session(lesson())
        await orchestrator.start(session.id)
        await orchestrator.submit_answer(session.id, StudentResponse(answer_text="yes because part of a whole"))

        with self.assertRaises(InvalidTransitionError):
            await orchestrator.submit_answer(session.id, StudentResponse(answer_text="yes again"))

    async def test_provider_failure_and_malformed_output_fall_back(self) -> None:
        for provider in (FailingProvider(), MalformedProvider()):
            orchestrator = await self.make_orchestrator(provider=provider)
            session = await orchestrator.create_session(lesson())
            started = await orchestrator.start(session.id)
            self.assertEqual(started.state, "waiting_for_answer")
            self.assertIn("Fractions", started.turns[0].text)

    async def test_provider_timeout_falls_back(self) -> None:
        orchestrator = await self.make_orchestrator(
            provider=SlowProvider(),
            config=TeachingConfig(provider_timeout_seconds=0.001, structured_output_retries=0),
        )
        session = await orchestrator.create_session(lesson())
        started = await orchestrator.start(session.id)

        self.assertEqual(started.state, "waiting_for_answer")
        self.assertIn("Fractions", started.turns[0].text)

    async def test_session_persistence_and_events(self) -> None:
        orchestrator = await self.make_orchestrator()
        session = await orchestrator.create_session(lesson())
        await orchestrator.start(session.id)

        restored = await orchestrator.get_session(session.id)
        events = orchestrator.events(session.id)

        self.assertEqual(restored.id, session.id)
        self.assertGreaterEqual(len(events), 4)
        self.assertEqual(events[-1]["state"], "waiting_for_answer")


class TestTeachingApi(unittest.TestCase):
    def test_api_lesson_flow_and_error_responses(self) -> None:
        app = create_app(build_application_runtime(runtime_config()))
        payload = lesson().model_dump()

        with TestClient(app) as client:
            created = client.post("/api/v1/teaching/sessions", json=payload)
            self.assertEqual(created.status_code, 200)
            session_id = created.json()["session"]["id"]

            invalid_answer = client.post(f"/api/v1/teaching/sessions/{session_id}/answer", json={"answer_text": "early"})
            self.assertEqual(invalid_answer.status_code, 409)

            started = client.post(f"/api/v1/teaching/sessions/{session_id}/start")
            self.assertEqual(started.status_code, 200)
            self.assertEqual(started.json()["session"]["state"], "waiting_for_answer")

            answer = client.post(f"/api/v1/teaching/sessions/{session_id}/answer", json={"answer_text": "yes because part of a whole"})
            self.assertEqual(answer.status_code, 200)
            self.assertEqual(answer.json()["session"]["state"], "session_complete")

            summary = client.get(f"/api/v1/teaching/sessions/{session_id}/summary")
            self.assertEqual(summary.status_code, 200)
            self.assertIn("Fractions", summary.json()["recap"])

            missing = client.get("/api/v1/teaching/sessions/missing")
            self.assertEqual(missing.status_code, 404)

    def test_api_validation_and_frontend_assets(self) -> None:
        app = create_app(build_application_runtime(runtime_config()))

        with TestClient(app) as client:
            invalid = client.post("/api/v1/teaching/sessions", json={"topic": ""})
            html = client.get("/")
            chat = client.post("/chat", json={"user_text": "hello", "retrieve_memory": False, "store_log": False})
            camera = client.get("/camera/status")

        self.assertEqual(invalid.status_code, 422)
        self.assertEqual(html.status_code, 200)
        self.assertIn("lesson-setup-form", html.text)
        self.assertIn("debug-dashboard", html.text)
        self.assertEqual(chat.status_code, 200)
        self.assertEqual(camera.status_code, 200)


if __name__ == "__main__":
    unittest.main()
