import base64
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from pydantic import ValidationError

from api.main import create_app
from server.config import DEFAULT_CONFIG_PATH, RuntimeConfig, _deep_merge, _read_yaml
from server.runtime import build_application_runtime


DEMO_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "laptop-demo.yaml"


def laptop_config(db_path: Path) -> RuntimeConfig:
    raw = _deep_merge(_read_yaml(DEFAULT_CONFIG_PATH), _read_yaml(DEMO_CONFIG_PATH))
    raw["database"] = {"sqlite_path": str(db_path)}
    return RuntimeConfig.model_validate(raw)


class TestPhase8LaptopDemo(unittest.TestCase):
    def test_laptop_demo_config_uses_only_mock_local_providers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = laptop_config(Path(tmp) / "demo.db")

        self.assertEqual(cfg.environment, "laptop_demo")
        self.assertEqual(cfg.llm.active_provider, "mock")
        self.assertEqual(cfg.camera.provider, "mock")
        self.assertEqual(cfg.audio.input_provider, "mock")
        self.assertEqual(cfg.audio.output_provider, "mock")
        self.assertEqual(cfg.wake_word.provider, "mock")
        self.assertEqual(cfg.stt.provider, "mock")
        self.assertEqual(cfg.tts.provider, "mock")
        self.assertEqual(cfg.recognition.face_detection_provider, "mock")
        self.assertEqual(cfg.recognition.face_recognition_provider, "mock")
        self.assertEqual(cfg.engagement.provider, "mock")
        self.assertEqual(cfg.hardware.provider, "mock")
        self.assertTrue(cfg.hardware.enabled)
        self.assertFalse(cfg.hardware.physical_output_enabled)
        self.assertTrue(cfg.feature_flags.hardware_control)

    def test_physical_output_cannot_be_enabled_without_approved_profile(self) -> None:
        raw = _deep_merge(_read_yaml(DEFAULT_CONFIG_PATH), _read_yaml(DEMO_CONFIG_PATH))
        raw["hardware"] = copy.deepcopy(raw["hardware"])
        raw["hardware"]["provider"] = "esp32"
        raw["hardware"]["transport"] = "serial"
        raw["hardware"]["physical_output_enabled"] = True

        with self.assertRaises(ValidationError):
            RuntimeConfig.model_validate(raw)

    def test_mock_end_to_end_release_workflow_and_privacy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict("os.environ", {"ADMIN_API_TOKEN": "demo-token"}, clear=False):
            runtime = build_application_runtime(laptop_config(Path(tmp) / "demo.db"))
            app = create_app(runtime)
            headers = {"X-Admin-Token": "demo-token"}
            with TestClient(app) as client:
                health = client.get("/api/v1/health")
                self.assertEqual(health.status_code, 200)
                body = health.json()
                self.assertEqual(body["runtime"]["environment"], "laptop_demo")
                self.assertFalse(body["runtime"]["hardware"]["physical_output_enabled"])

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
                        json={"embedding_seed": "phase8-asha", "pose_label": pose, "quality_override": "ok"},
                    )
                    self.assertEqual(sample.status_code, 200)
                    self.assertTrue(sample.json()["accepted"])
                completed = client.post(f"/api/v1/admin/registrations/{registration_id}/complete", headers=headers)
                self.assertEqual(completed.status_code, 200)
                self.assertTrue(completed.json()["verified"])

                recognized = client.post("/api/v1/student/recognize", json={"embedding_seed": "phase8-asha"})
                self.assertEqual(recognized.status_code, 200)
                self.assertEqual(recognized.json()["display_name"], "Asha")

                created = client.post(
                    "/api/v1/teaching/sessions",
                    json={
                        "student_display_name": "Asha",
                        "grade_level": "6",
                        "topic": "Fractions",
                        "language": "en",
                        "objective": "Understand fractions.",
                    },
                )
                self.assertEqual(created.status_code, 200)
                session_id = created.json()["session"]["id"]
                self.assertEqual(client.post(f"/api/v1/teaching/sessions/{session_id}/start").status_code, 200)
                voice = client.post("/api/v1/audio/push-to-talk/start", json={"session_id": session_id})
                self.assertEqual(voice.status_code, 200)
                self.assertTrue(voice.json()["ok"])
                duplicate = client.post(
                    f"/api/v1/teaching/sessions/{session_id}/answer",
                    json={"answer_text": "A second duplicate answer."},
                )
                self.assertEqual(duplicate.status_code, 409)
                summary = client.get(f"/api/v1/teaching/sessions/{session_id}/summary")
                self.assertEqual(summary.status_code, 200)

                second = client.post(
                    "/api/v1/teaching/sessions",
                    json={
                        "student_display_name": "Asha",
                        "grade_level": "6",
                        "topic": "Geometry",
                        "language": "en",
                        "objective": "Recognize shapes.",
                    },
                )
                second_id = second.json()["session"]["id"]
                client.post(f"/api/v1/teaching/sessions/{second_id}/start")
                for index in range(3):
                    engagement = client.post(
                        f"/api/v1/engagement/sessions/{second_id}/signals",
                        headers=headers,
                        json={"timestamp_utc": f"2026-07-28T00:00:0{index}+00:00", "unclear_answer_count": 2},
                    )
                self.assertEqual(engagement.status_code, 200)
                self.assertIn(engagement.json()["state"], {"question_repeat", "gentle_prompt"})

                chat = client.post("/chat", json={"user_text": "hello", "retrieve_memory": False, "store_log": False})
                self.assertEqual(chat.status_code, 200)
                tts = client.post("/api/v1/audio/tts/start", json={"text": "hello"})
                self.assertEqual(tts.status_code, 200)
                wake = client.post("/api/v1/audio/wake-word/activate")
                self.assertEqual(wake.status_code, 200)

                camera_status = client.get("/camera/status")
                self.assertEqual(camera_status.status_code, 200)

                action = client.post("/api/v1/hardware/actions", headers=headers, json={"action": "small_nod"})
                self.assertEqual(action.status_code, 200)
                estop = client.post("/api/v1/hardware/emergency-stop", headers=headers)
                self.assertEqual(estop.status_code, 200)
                rejected = client.post("/api/v1/hardware/actions", headers=headers, json={"action": "neutral"})
                self.assertEqual(rejected.status_code, 409)
                reset = client.post("/api/v1/hardware/emergency-stop/reset", headers=headers, json={"confirm": True})
                self.assertEqual(reset.status_code, 200)

                payload_text = str(
                    {
                        "student": recognized.json(),
                        "engagement": engagement.json(),
                        "hardware": client.get("/api/v1/hardware/health", headers=headers).json(),
                    }
                ).lower()
                for blocked in ("embedding", "api_key", "admin_api_token", "raw confidence", "serial_port"):
                    self.assertNotIn(blocked, payload_text)

    def test_sqlite_recovery_after_application_restart(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "recovery.db"
            runtime_one = build_application_runtime(laptop_config(db_path))
            app_one = create_app(runtime_one)
            with TestClient(app_one) as client:
                created = client.post(
                    "/api/v1/teaching/sessions",
                    json={
                        "student_display_name": "Asha",
                        "grade_level": "6",
                        "topic": "Fractions",
                        "language": "en",
                        "objective": "Understand fractions.",
                    },
                )
                session_id = created.json()["session"]["id"]
                client.post(f"/api/v1/teaching/sessions/{session_id}/start")

            runtime_two = build_application_runtime(laptop_config(db_path))
            app_two = create_app(runtime_two)
            with TestClient(app_two) as client:
                recovered = client.get(f"/api/v1/teaching/sessions/{session_id}")
                self.assertEqual(recovered.status_code, 200)
                self.assertEqual(recovered.json()["session"]["state"], "waiting_for_answer")


if __name__ == "__main__":
    unittest.main()
