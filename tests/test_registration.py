import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from api.main import create_app
from server.config import RecognitionConfig, RuntimeConfig
from server.face_registration import (
    FaceRegistrationService,
    RecognitionRequest,
    RegistrationError,
    RegistrationSample,
    RegistrationStart,
)
from server.runtime import build_application_runtime
from server.student_store import SQLiteStudentRepository


def runtime_config(db_path: Path) -> RuntimeConfig:
    return RuntimeConfig.model_validate(
        {
            "llm": {"active_provider": "mock", "profiles": {"mock": {"provider": "mock", "model": "mock"}}},
            "camera": {"provider": "mock", "width": 160, "height": 120},
            "recognition": {
                "face_detection_provider": "mock",
                "face_recognition_provider": "mock",
                "registration_sample_count": 3,
                "recognition_interval_seconds": 0.001,
            },
            "database": {"sqlite_path": str(db_path)},
            "feature_flags": {
                "student_registration": True,
                "face_recognition": True,
                "student_ui": True,
                "text_input": True,
            },
        }
    )


class TestSQLiteStudentRegistration(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "students.db"
        self.store = SQLiteStudentRepository(str(self.db_path))
        self.service = FaceRegistrationService(
            self.store,
            RecognitionConfig(
                registration_sample_count=3,
                recognition_interval_seconds=0.001,
                face_match_threshold=0.72,
            ),
        )

    async def asyncTearDown(self) -> None:
        self.tmp.cleanup()

    async def start_valid_registration(self):
        result = await self.service.start_registration(
            RegistrationStart(display_name="Asha", grade_level="6", language="en", consent_given=True)
        )
        return result["registration"]["id"], result["student"]["id"]

    async def complete_valid_registration(self, seed: str = "asha-face"):
        registration_id, student_id = await self.start_valid_registration()
        for pose in ("center", "left", "right"):
            sample = await self.service.submit_sample(
                registration_id,
                RegistrationSample(embedding_seed=seed, pose_label=pose, quality_override="ok"),
            )
            self.assertTrue(sample["accepted"])
        completed = await self.service.complete_registration(registration_id)
        self.assertTrue(completed["verified"])
        return registration_id, student_id

    async def test_sqlite_initialization_and_persistence(self) -> None:
        await self.store.initialize()
        student = await self.store.create_student("Ravi", "5", "en", True, status="registered")
        fetched = await self.store.get_student(student["id"])

        self.assertTrue(self.db_path.exists())
        self.assertEqual(fetched["display_name"], "Ravi")
        self.assertTrue(fetched["consent_given"])

    async def test_consent_is_required(self) -> None:
        with self.assertRaises(RegistrationError):
            await self.service.start_registration(RegistrationStart(display_name="Asha", consent_given=False))

    async def test_valid_registration_and_temporary_frame_cleanup(self) -> None:
        _, student_id = await self.complete_valid_registration()
        student = await self.store.get_student(student_id)
        embeddings = await self.store.get_embeddings(student_id)

        self.assertEqual(student["registration_status"], "registered")
        self.assertEqual(len(embeddings), 3)
        self.assertEqual(self.service.temporary_frames_retained, 0)

    async def test_rejects_dark_blurry_no_face_and_multi_face_samples(self) -> None:
        registration_id, _ = await self.start_valid_registration()
        for override, reason in (
            ("dark", "dark_or_overexposed"),
            ("blurry", "blurry"),
            ("no_face", "no_face"),
            ("multi_face", "multi_face"),
        ):
            result = await self.service.submit_sample(
                registration_id,
                RegistrationSample(embedding_seed=override, quality_override=override),
            )
            self.assertFalse(result["accepted"])
            self.assertEqual(result["reason"], reason)

    async def test_recognition_above_and_below_threshold_and_guest_behavior(self) -> None:
        await self.complete_valid_registration(seed="asha-face")
        matched = await self.service.recognize_current_student(RecognitionRequest(embedding_seed="asha-face"))
        await asyncio.sleep(0.01)
        unknown = await self.service.recognize_current_student(RecognitionRequest(embedding_seed="different-face"))

        self.assertTrue(matched["recognized"])
        self.assertEqual(matched["display_name"], "Asha")
        self.assertFalse(unknown["recognized"])
        self.assertEqual(unknown["display_name"], "Guest")

    async def test_duplicate_name_profile_deletion_and_registration_cancellation(self) -> None:
        registration_id, student_id = await self.start_valid_registration()

        with self.assertRaises(RegistrationError):
            await self.service.start_registration(RegistrationStart(display_name="asha", consent_given=True))

        cancelled = await self.service.cancel_registration(registration_id)
        self.assertTrue(cancelled["cancelled"])
        self.assertIsNone(await self.store.get_student(student_id))

        _, registered_id = await self.complete_valid_registration(seed="new-asha")
        self.assertTrue(await self.store.delete_student(registered_id))
        self.assertEqual(await self.store.get_embeddings(registered_id), [])

    async def test_mock_end_to_end_workflow(self) -> None:
        registration_id, _ = await self.complete_valid_registration(seed="mock-e2e")
        status = await self.service.registration_status(registration_id)

        self.assertEqual(status["registration"]["status"], "completed")
        self.assertEqual(status["student"]["registration_status"], "registered")


class TestRegistrationApi(unittest.TestCase):
    def test_api_authorization_validation_and_existing_regressions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict("os.environ", {"ADMIN_API_TOKEN": "secret"}, clear=False):
            app = create_app(build_application_runtime(runtime_config(Path(tmp) / "students.db")))
            with TestClient(app) as client:
                unauthorized = client.get("/api/v1/admin/students")
                self.assertEqual(unauthorized.status_code, 403)

                headers = {"X-Admin-Token": "secret"}
                missing_consent = client.post(
                    "/api/v1/admin/registrations",
                    headers=headers,
                    json={"display_name": "Asha", "consent_given": False},
                )
                self.assertEqual(missing_consent.status_code, 400)

                started = client.post(
                    "/api/v1/admin/registrations",
                    headers=headers,
                    json={"display_name": "Asha", "grade_level": "6", "language": "en", "consent_given": True},
                )
                self.assertEqual(started.status_code, 200)
                registration_id = started.json()["registration"]["id"]

                for pose in ("center", "left", "right"):
                    sample = client.post(
                        f"/api/v1/admin/registrations/{registration_id}/samples",
                        headers=headers,
                        json={"embedding_seed": "asha-api", "pose_label": pose, "quality_override": "ok"},
                    )
                    self.assertEqual(sample.status_code, 200)
                    self.assertTrue(sample.json()["accepted"])

                completed = client.post(f"/api/v1/admin/registrations/{registration_id}/complete", headers=headers)
                self.assertEqual(completed.status_code, 200)
                self.assertTrue(completed.json()["verified"])
                self.assertNotIn("embedding", str(completed.json()).lower())

                recognized = client.post(
                    "/api/v1/admin/recognize",
                    headers=headers,
                    json={"embedding_seed": "asha-api", "quality_override": "ok"},
                )
                self.assertEqual(recognized.status_code, 200)
                self.assertEqual(recognized.json()["display_name"], "Asha")

                students = client.get("/api/v1/admin/students", headers=headers)
                student_id = students.json()["students"][0]["id"]
                delete_without_confirm = client.delete(f"/api/v1/admin/students/{student_id}", headers=headers)
                self.assertEqual(delete_without_confirm.status_code, 422)
                deleted = client.delete(f"/api/v1/admin/students/{student_id}?confirm=true", headers=headers)
                self.assertEqual(deleted.status_code, 200)

                teaching = client.post(
                    "/api/v1/teaching/sessions",
                    json={
                        "student_display_name": "Guest",
                        "grade_level": "6",
                        "topic": "Fractions",
                        "language": "en",
                        "objective": "Understand fractions.",
                    },
                )
                chat = client.post("/chat", json={"user_text": "hello", "retrieve_memory": False, "store_log": False})
                camera = client.get("/camera/status")

        self.assertEqual(teaching.status_code, 200)
        self.assertEqual(chat.status_code, 200)
        self.assertEqual(camera.status_code, 200)


if __name__ == "__main__":
    unittest.main()
