import asyncio
import inspect
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi.testclient import TestClient
from pydantic import ValidationError

from api.main import create_app
from server.config import HardwareConfig, RuntimeConfig
from server.engagement import EngagementRuntime
from server.hardware import (
    PROTOCOL_VERSION,
    HardwareCommand,
    HardwareRuntime,
    SimulatedESP32Transport,
    command_for_action,
)
from server.runtime import build_application_runtime


def ts(offset_seconds: float = 0) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=offset_seconds)).isoformat()


def runtime_config(db_path: Path, hardware_extra: dict | None = None) -> RuntimeConfig:
    hardware = {
        "enabled": True,
        "provider": "mock",
        "transport": "mock",
        "physical_output_enabled": False,
        "unsafe_operating_zone_clear": True,
        "motion_cooldown_seconds": 0,
        "retry_limit": 1,
        "command_timeout_seconds": 0.05,
        "stale_command_seconds": 1,
        "servo_limits": {
            "min_angle_deg": -20,
            "max_angle_deg": 20,
            "max_speed_deg_per_second": 30,
            "max_duration_seconds": 1,
        },
        "motor_limits": {"max_speed_percent": 0, "max_duration_seconds": 0},
    }
    if hardware_extra:
        for key, value in hardware_extra.items():
            if isinstance(value, dict) and isinstance(hardware.get(key), dict):
                hardware[key].update(value)
            else:
                hardware[key] = value
    return RuntimeConfig.model_validate(
        {
            "llm": {"active_provider": "mock", "profiles": {"mock": {"provider": "mock", "model": "mock"}}},
            "camera": {"provider": "mock", "width": 160, "height": 120},
            "audio": {"input_provider": "mock", "output_provider": "mock", "min_recording_ms": 0},
            "wake_word": {"provider": "mock"},
            "vad": {"provider": "mock", "min_recording_ms": 0},
            "stt": {"provider": "mock"},
            "tts": {"provider": "mock"},
            "database": {"sqlite_path": str(db_path)},
            "hardware": hardware,
            "feature_flags": {
                "student_ui": True,
                "backend_camera_stream": True,
                "text_input": True,
                "hardware_control": True,
            },
        }
    )


class TestHardwareRuntime(unittest.IsolatedAsyncioTestCase):
    async def test_mock_lifecycle_valid_ack_and_history(self) -> None:
        runtime = HardwareRuntime(HardwareConfig(enabled=True, unsafe_operating_zone_clear=True, motion_cooldown_seconds=0))
        await runtime.start()
        ack = await runtime.submit(command_for_action("small_nod", runtime.config))
        await runtime.stop()

        self.assertTrue(ack.ok)
        self.assertEqual(ack.status, "ack")
        self.assertEqual(runtime.state, "safe_stopped")
        self.assertTrue(runtime.history())

    async def test_invalid_out_of_range_duplicate_and_stale_rejection(self) -> None:
        runtime = HardwareRuntime(HardwareConfig(enabled=True, unsafe_operating_zone_clear=True, motion_cooldown_seconds=0))
        await runtime.start()
        out_of_range = HardwareCommand("angle", PROTOCOL_VERSION, ts(), "small_nod", {"angle_deg": 90})
        stale = HardwareCommand("stale", PROTOCOL_VERSION, ts(-20), "small_nod", {})
        valid = command_for_action("neutral", runtime.config)
        first = await runtime.submit(valid)
        duplicate = await runtime.submit(valid)
        unsupported = await runtime.submit(HardwareCommand("bad", PROTOCOL_VERSION, ts(), "dance", {}))
        await runtime.stop()

        self.assertFalse((await runtime.submit(out_of_range)).ok)
        self.assertFalse((await runtime.submit(stale)).ok)
        self.assertTrue(first.ok)
        self.assertFalse(duplicate.ok)
        self.assertFalse(unsupported.ok)

    async def test_timeout_retry_limit_and_lost_connection_safe_stop(self) -> None:
        transport = SimulatedESP32Transport()
        runtime = HardwareRuntime(
            HardwareConfig(
                enabled=True,
                unsafe_operating_zone_clear=True,
                command_timeout_seconds=0.001,
                retry_limit=1,
                motion_cooldown_seconds=0,
            ),
            transport=transport,
        )
        await runtime.start()
        transport.drop_next = True
        transport.delay_seconds = 0.01
        ack = await runtime.submit(command_for_action("small_head_turn", runtime.config))
        await runtime.mark_connection_lost()

        self.assertFalse(ack.ok)
        self.assertEqual(ack.status, "timeout")
        self.assertEqual(runtime.state, "emergency_stopped")

    async def test_emergency_stop_is_idempotent_and_bypasses_transport_failures(self) -> None:
        transport = SimulatedESP32Transport()
        runtime = HardwareRuntime(
            HardwareConfig(
                enabled=True,
                unsafe_operating_zone_clear=True,
                motion_cooldown_seconds=0,
                command_timeout_seconds=0.001,
            ),
            transport=transport,
        )
        await runtime.start()
        cancel = await runtime.cancel()
        transport.drop_next = True
        transport.delay_seconds = 0.01
        estop = await runtime.emergency_stop()
        repeated = await runtime.emergency_stop()
        rejected = await runtime.submit(command_for_action("neutral", runtime.config))

        self.assertTrue(cancel.ok)
        self.assertTrue(estop.ok)
        self.assertEqual(estop.status, "emergency_stopped")
        self.assertTrue(repeated.ok)
        self.assertEqual(repeated.status, "emergency_stopped")
        self.assertEqual(runtime.state, "emergency_stopped")
        self.assertFalse(rejected.ok)

    async def test_only_explicit_reset_clears_emergency_stop(self) -> None:
        runtime = HardwareRuntime(HardwareConfig(enabled=True, unsafe_operating_zone_clear=True, motion_cooldown_seconds=0))
        await runtime.start()
        await runtime.emergency_stop()
        cancelled = await runtime.cancel()
        still_rejected = await runtime.submit(command_for_action("neutral", runtime.config))
        denied_reset = await runtime.reset_emergency_stop(confirm=False)
        accepted_reset = await runtime.reset_emergency_stop(confirm=True)
        allowed_after_reset = await runtime.submit(command_for_action("neutral", runtime.config))

        self.assertTrue(cancelled.ok)
        self.assertEqual(cancelled.status, "emergency_stopped")
        self.assertFalse(still_rejected.ok)
        self.assertFalse(denied_reset.ok)
        self.assertTrue(accepted_reset.ok)
        self.assertEqual(accepted_reset.status, "emergency_stop_reset")
        self.assertTrue(allowed_after_reset.ok)

    async def test_physical_output_disabled_for_esp32(self) -> None:
        runtime = HardwareRuntime(
            HardwareConfig(enabled=True, provider="esp32", transport="serial", unsafe_operating_zone_clear=True)
        )
        ack = await runtime.submit(command_for_action("neutral", runtime.config))

        self.assertFalse(ack.ok)
        self.assertEqual(ack.status, "physical_output_disabled")

    async def test_physical_profile_validation_blocks_unknown_hardware(self) -> None:
        with self.assertRaises(ValidationError):
            HardwareConfig(
                enabled=True,
                provider="esp32",
                transport="serial",
                physical_output_enabled=True,
                hardware_profile_approved=False,
            )


class TestHardwareApi(unittest.TestCase):
    def test_api_lifecycle_privacy_and_regression(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app_runtime = build_application_runtime(runtime_config(Path(tmp) / "hardware.db"))
            app = create_app(app_runtime)
            with TestClient(app) as client:
                health = client.get("/api/v1/hardware/health")
                actions = client.get("/api/v1/hardware/actions")
                command = client.post("/api/v1/hardware/actions", json={"action": "small_nod"})
                history = client.get("/api/v1/hardware/history")
                chat = client.post("/chat", json={"user_text": "hello", "store_log": False})
                camera = client.get("/camera/status")

        self.assertEqual(health.status_code, 200)
        self.assertEqual(actions.status_code, 200)
        self.assertEqual(command.status_code, 200)
        self.assertEqual(history.status_code, 200)
        self.assertEqual(chat.status_code, 200)
        self.assertEqual(camera.status_code, 200)
        safe_payload = str(health.json()) + str(history.json())
        self.assertNotIn("COM", safe_payload)
        self.assertNotIn("serial_port", safe_payload)
        self.assertNotIn("Traceback", safe_payload)

    def test_api_cancel_emergency_and_pause_stop_cancel_motion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app_runtime = build_application_runtime(runtime_config(Path(tmp) / "hardware.db"))
            app = create_app(app_runtime)
            with TestClient(app) as client:
                cancel = client.post("/api/v1/hardware/cancel")
                estop = client.post("/api/v1/hardware/emergency-stop")
                denied_reset = client.post("/api/v1/hardware/emergency-stop/reset", json={"confirm": False})
                reset = client.post("/api/v1/hardware/emergency-stop/reset", json={"confirm": True})
                created = client.post(
                    "/api/v1/teaching/sessions",
                    json={
                        "student_display_name": "Asha",
                        "grade_level": "grade_6",
                        "topic": "Fractions",
                        "language": "en",
                        "objective": "Understand fractions.",
                    },
                ).json()
                session_id = created["session"]["id"]
                client.post(f"/api/v1/teaching/sessions/{session_id}/start")
                pause = client.post(f"/api/v1/teaching/sessions/{session_id}/pause")
                stop = client.post(f"/api/v1/teaching/sessions/{session_id}/stop")

        self.assertEqual(cancel.status_code, 200)
        self.assertEqual(estop.status_code, 200)
        self.assertEqual(denied_reset.status_code, 200)
        self.assertFalse(denied_reset.json()["ack"]["ok"])
        self.assertEqual(reset.status_code, 200)
        self.assertTrue(reset.json()["ack"]["ok"])
        self.assertEqual(pause.status_code, 200)
        self.assertEqual(stop.status_code, 200)

    def test_disabled_by_default_in_config(self) -> None:
        cfg = RuntimeConfig.model_validate(
            {"llm": {"active_provider": "mock", "profiles": {"mock": {"provider": "mock", "model": "mock"}}}}
        )

        self.assertFalse(cfg.hardware.enabled)
        self.assertFalse(cfg.hardware.physical_output_enabled)
        self.assertFalse(cfg.feature_flags.hardware_control)

    def test_no_direct_llm_or_engagement_to_motion_path(self) -> None:
        engagement_source = inspect.getsource(EngagementRuntime)

        self.assertNotIn("hardware", engagement_source.lower())
        self.assertNotIn("submit_predefined_action", engagement_source)


if __name__ == "__main__":
    unittest.main()
