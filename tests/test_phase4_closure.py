import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

from api.main import create_app
from server.config import ConfigurationError, RecognitionConfig, RuntimeConfig
from server.face_providers import OpenCVYuNetSFaceProvider, normalize_embedding
from server.runtime import build_application_runtime
from server.teaching import LessonConfig


def closure_config(db_path: Path, extra: dict | None = None) -> RuntimeConfig:
    raw = {
        "bind_host": "127.0.0.1",
        "llm": {"active_provider": "mock", "profiles": {"mock": {"provider": "mock", "model": "mock"}}},
        "camera": {"provider": "mock", "width": 160, "height": 120},
        "recognition": {"registration_sample_count": 3, "recognition_interval_seconds": 0.001},
        "database": {"sqlite_path": str(db_path)},
        "feature_flags": {
            "student_registration": True,
            "face_recognition": True,
            "backend_camera_stream": True,
            "student_ui": True,
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


class TestLocalFaceProviderClosure(unittest.TestCase):
    def test_real_provider_configuration_missing_model_fails_startup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = closure_config(
                Path(tmp) / "students.db",
                {
                    "recognition": {
                        "face_detection_provider": "opencv",
                        "face_recognition_provider": "local",
                        "face_detection_model_path": str(Path(tmp) / "missing-yunet.onnx"),
                        "face_recognition_model_path": str(Path(tmp) / "missing-sface.onnx"),
                    }
                },
            )
            app = create_app(build_application_runtime(cfg))
            with self.assertRaises(RuntimeError):
                with TestClient(app):
                    pass

    def test_face_alignment_and_embedding_normalisation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            detector = Path(tmp) / "yunet.onnx"
            recognizer = Path(tmp) / "sface.onnx"
            detector.write_bytes(b"model")
            recognizer.write_bytes(b"model")
            provider = OpenCVYuNetSFaceProvider(
                RecognitionConfig(
                    face_detection_provider="opencv",
                    face_recognition_provider="local",
                    face_detection_model_path=str(detector),
                    face_recognition_model_path=str(recognizer),
                )
            )

            class FakeRecognizer:
                def alignCrop(self, image, face):  # noqa: N802
                    return image[0:2, 0:2]

                def feature(self, aligned):
                    return np.array([[3.0, 4.0]], dtype=np.float32)

            provider.recognizer = FakeRecognizer()
            aligned = provider.align_face(np.ones((4, 4, 3), dtype=np.uint8), np.zeros((15,), dtype=np.float32))
            embedding = provider.embed_aligned(aligned)

        self.assertEqual(aligned.shape, (2, 2, 3))
        self.assertAlmostEqual(math.sqrt(sum(value * value for value in embedding)), 1.0, places=6)
        self.assertAlmostEqual(math.sqrt(sum(value * value for value in normalize_embedding([10.0, 0.0]))), 1.0, places=6)

    def test_lan_startup_rejected_without_admin_token(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict("os.environ", {"ADMIN_API_TOKEN": ""}, clear=False):
            cfg = closure_config(Path(tmp) / "students.db", {"bind_host": "0.0.0.0"})
            with self.assertRaises(ConfigurationError):
                build_application_runtime(cfg)


class TestSQLiteTeachingClosure(unittest.TestCase):
    def lesson(self) -> LessonConfig:
        return LessonConfig(
            student_display_name="Asha",
            grade_level="6",
            topic="Fractions",
            language="en",
            objective="Understand fractions.",
        )

    def test_sqlite_teaching_session_reload_and_restart_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "students.db"
            runtime1 = build_application_runtime(closure_config(db_path))
            with TestClient(create_app(runtime1)):
                session = runtime1.teaching.create_session
                created = __import__("asyncio").run(session(self.lesson()))
                started = __import__("asyncio").run(runtime1.teaching.start(created.id))
                self.assertEqual(started.state, "waiting_for_answer")

            runtime2 = build_application_runtime(closure_config(db_path))
            with TestClient(create_app(runtime2)):
                recovered = __import__("asyncio").run(runtime2.teaching.recover_active_sessions())
                restored = __import__("asyncio").run(runtime2.teaching.get_session(created.id))

        self.assertEqual(recovered[0].id, created.id)
        self.assertEqual(restored.state, "waiting_for_answer")
        self.assertEqual(restored.config.topic, "Fractions")

    def test_duplicate_answer_protection_after_restart(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "students.db"
            runtime1 = build_application_runtime(closure_config(db_path))
            with TestClient(create_app(runtime1)):
                created = __import__("asyncio").run(runtime1.teaching.create_session(self.lesson()))
                __import__("asyncio").run(runtime1.teaching.start(created.id))
                __import__("asyncio").run(
                    runtime1.teaching.submit_answer(
                        created.id,
                        __import__("server.teaching", fromlist=["StudentResponse"]).StudentResponse(
                            answer_text="A fraction is part of a whole"
                        ),
                    )
                )

            runtime2 = build_application_runtime(closure_config(db_path))
            with TestClient(create_app(runtime2)) as client:
                duplicate = client.post(
                    f"/api/v1/teaching/sessions/{created.id}/answer",
                    json={"answer_text": "same answer again"},
                )

        self.assertEqual(duplicate.status_code, 409)

    def test_schema_initialization_and_migration_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = build_application_runtime(closure_config(Path(tmp) / "students.db"))
            with TestClient(create_app(runtime)):
                version = __import__("asyncio").run(runtime.student_store.schema_version())

        self.assertGreaterEqual(version, 2)


if __name__ == "__main__":
    unittest.main()
