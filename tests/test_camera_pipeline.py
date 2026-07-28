import asyncio
import unittest

from fastapi.testclient import TestClient
from pydantic import ValidationError

from api.main import create_app
from server.camera_pipeline import FramePipeline, MockCameraDriver, Picamera2CameraDriver, Picamera2UnavailableError
from server.config import CameraConfig, RuntimeConfig
from server.runtime import build_application_runtime


def runtime_config(camera_provider: str = "mock") -> RuntimeConfig:
    return RuntimeConfig.model_validate(
        {
            "llm": {"active_provider": "mock", "profiles": {"mock": {"provider": "mock", "model": "mock"}}},
            "camera": {
                "provider": camera_provider,
                "width": 160,
                "height": 120,
                "preview_fps": 12,
                "analysis_fps": 5,
                "jpeg_quality": 70,
            },
            "feature_flags": {
                "browser_camera": camera_provider == "browser",
                "backend_camera_stream": camera_provider in {"mock", "picamera2"},
            },
        }
    )


class TestCameraPipeline(unittest.IsolatedAsyncioTestCase):
    async def test_mock_camera_start_frame_and_shutdown(self) -> None:
        config = CameraConfig(provider="mock", width=160, height=120, preview_fps=12)
        driver = MockCameraDriver(config)
        pipeline = FramePipeline(driver, config)

        await pipeline.start()
        frame = await pipeline.wait_for_frame(timeout=2)
        await pipeline.stop()

        self.assertIsNotNone(frame)
        self.assertEqual(frame.width, 160)
        self.assertEqual(frame.height, 120)
        self.assertGreater(frame.sequence, 0)
        self.assertEqual(pipeline.status().state, "stopped")

    async def test_latest_frame_replaces_stale_frames(self) -> None:
        config = CameraConfig(provider="mock", width=160, height=120, preview_fps=20)
        pipeline = FramePipeline(MockCameraDriver(config), config)

        await pipeline.start()
        first = await pipeline.wait_for_frame(timeout=2)
        await asyncio.sleep(0.2)
        latest = pipeline.latest_frame
        await pipeline.stop()

        self.assertIsNotNone(first)
        self.assertIsNotNone(latest)
        self.assertGreater(latest.sequence, first.sequence)

    async def test_multiple_preview_clients_share_one_camera_owner(self) -> None:
        config = CameraConfig(provider="mock", width=160, height=120, preview_fps=12)
        driver = MockCameraDriver(config)
        pipeline = FramePipeline(driver, config)

        await pipeline.start()
        stream_a = pipeline.mjpeg_frames()
        stream_b = pipeline.mjpeg_frames()
        chunk_a = await anext(stream_a)
        chunk_b = await anext(stream_b)
        clients = pipeline.preview_clients
        await stream_a.aclose()
        await stream_b.aclose()
        await pipeline.stop()

        self.assertTrue(chunk_a.startswith(b"--frame\r\n"))
        self.assertTrue(chunk_b.startswith(b"--frame\r\n"))
        self.assertEqual(driver.sequence, pipeline.latest_frame.sequence)
        self.assertEqual(clients, 2)

    async def test_status_event_generation(self) -> None:
        config = CameraConfig(provider="mock", width=160, height=120, preview_fps=12)
        pipeline = FramePipeline(MockCameraDriver(config), config)

        await pipeline.start()
        await pipeline.wait_for_frame(timeout=2)
        payload = pipeline.sse_payload()
        await pipeline.stop()

        self.assertIn("event: camera_status", payload)
        self.assertIn('"state": "ready"', payload)

    async def test_picamera2_unavailable_on_windows(self) -> None:
        config = CameraConfig(provider="picamera2")
        driver = Picamera2CameraDriver(config)

        with self.assertRaises(Picamera2UnavailableError):
            await driver.start()

    async def test_invalid_camera_configuration(self) -> None:
        with self.assertRaises(ValidationError):
            CameraConfig(provider="mock", preview_fps=0)


class TestCameraApi(unittest.TestCase):
    def test_mjpeg_response_format(self) -> None:
        asyncio.run(self._assert_mjpeg_response_format())

    async def _assert_mjpeg_response_format(self) -> None:
        config = CameraConfig(provider="mock", width=160, height=120, preview_fps=12)
        pipeline = FramePipeline(MockCameraDriver(config), config)

        await pipeline.start()
        stream = pipeline.mjpeg_frames()
        first_chunk = await anext(stream)
        await stream.aclose()
        await pipeline.stop()

        self.assertTrue(first_chunk.startswith(b"--frame\r\n"))
        self.assertIn(b"Content-Type: image/jpeg", first_chunk)
        self.assertIn(b"\xff\xd8", first_chunk)

    def test_health_reports_camera_state_and_shutdown_cleans_up(self) -> None:
        runtime = build_application_runtime(runtime_config("mock"))
        app = create_app(runtime)

        with TestClient(app) as client:
            payload = client.get("/api/v1/health").json()
            self.assertEqual(payload["runtime"]["camera_provider"], "mock")
            self.assertIn(payload["runtime"]["camera_state"], {"starting", "ready"})

        self.assertEqual(runtime.camera_pipeline.status().state, "stopped")

    def test_status_endpoint(self) -> None:
        runtime = build_application_runtime(runtime_config("mock"))
        app = create_app(runtime)

        with TestClient(app) as client:
            payload = client.get("/camera/status").json()

        self.assertEqual(payload["provider"], "mock")
        self.assertIn(payload["state"], {"starting", "ready", "stopped"})

    def test_existing_browser_camera_endpoint_still_accepts_aliases(self) -> None:
        runtime = build_application_runtime(runtime_config("browser"))
        app = create_app(runtime)

        with TestClient(app) as client:
            response = client.post("/vision/track", json={"frame": "data:image/jpeg;base64,not-valid-base64"})

        self.assertEqual(response.status_code, 400)
        self.assertIn("Invalid base64 image payload", response.text)


if __name__ == "__main__":
    unittest.main()
