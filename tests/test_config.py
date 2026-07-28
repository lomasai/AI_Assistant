import tempfile
import unittest
from pathlib import Path

from server.config import ConfigurationError, RuntimeConfig, load_runtime_config


class TestRuntimeConfig(unittest.TestCase):
    def test_default_config_loads(self) -> None:
        config = load_runtime_config(device_path=Path("missing-device.yaml"))

        self.assertIsInstance(config, RuntimeConfig)
        self.assertIn("mock", config.llm.profiles)
        self.assertIn("groq", config.llm.profiles)
        self.assertEqual(config.camera.width, 640)

    def test_device_overlay_changes_active_provider(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            default_path = Path(tmp) / "default.yaml"
            device_path = Path(tmp) / "device.yaml"
            default_path.write_text(
                """
environment: test
llm:
  active_provider: mock
  profiles:
    mock:
      provider: mock
      model: mock
camera:
  provider: browser
""",
                encoding="utf-8",
            )
            device_path.write_text(
                """
llm:
  active_provider: openai
  profiles:
    openai:
      provider: openai_compatible
      base_url: https://api.openai.com/v1
      model: gpt-test
      api_key_env: OPENAI_API_KEY
""",
                encoding="utf-8",
            )

            config = load_runtime_config(default_path=default_path, device_path=device_path)

        self.assertEqual(config.llm.active_provider, "openai")
        self.assertIn("mock", config.llm.profiles)
        self.assertIn("openai", config.llm.profiles)

    def test_invalid_active_provider_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            default_path = Path(tmp) / "default.yaml"
            default_path.write_text(
                """
llm:
  active_provider: groq
  profiles:
    mock:
      provider: mock
      model: mock
""",
                encoding="utf-8",
            )

            with self.assertRaises(ConfigurationError):
                load_runtime_config(default_path=default_path, device_path=Path(tmp) / "missing.yaml")

    def test_legacy_inference_fps_camera_key_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            default_path = Path(tmp) / "default.yaml"
            default_path.write_text(
                """
llm:
  active_provider: mock
  profiles:
    mock:
      provider: mock
      model: mock
camera:
  provider: mock
  inference_fps: 4
""",
                encoding="utf-8",
            )

            config = load_runtime_config(default_path=default_path, device_path=Path(tmp) / "missing.yaml")

        self.assertEqual(config.camera.analysis_fps, 4)

    def test_invalid_recognition_configuration_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            default_path = Path(tmp) / "default.yaml"
            default_path.write_text(
                """
llm:
  active_provider: mock
  profiles:
    mock:
      provider: mock
      model: mock
recognition:
  brightness_min: 250
  brightness_max: 20
""",
                encoding="utf-8",
            )

            with self.assertRaises(ConfigurationError):
                load_runtime_config(default_path=default_path, device_path=Path(tmp) / "missing.yaml")


if __name__ == "__main__":
    unittest.main()
