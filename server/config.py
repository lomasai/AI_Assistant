"""Central validated application configuration.

Configuration is loaded from config/default.yaml, optionally overlaid with
config/device.yaml, and secrets are resolved from environment variables only.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator


class ConfigurationError(Exception):
    """Raised when application configuration is missing or invalid."""


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"
DEVICE_CONFIG_PATH = PROJECT_ROOT / "config" / "device.yaml"


class LLMProviderConfig(BaseModel):
    """Runtime settings for one LLM provider profile."""

    model_config = ConfigDict(extra="forbid")

    provider: Literal["openai_compatible", "mock"] = "openai_compatible"
    base_url: str = ""
    model: str = "mock-teacher"
    api_key_env: str = ""
    timeout_seconds: float = Field(default=30.0, gt=0)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, gt=0)
    supports_tools: bool = False
    supports_vision: bool = False

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str, info: Any) -> str:
        provider = info.data.get("provider")
        if provider == "mock":
            return value.strip()
        clean = value.strip()
        if not clean.startswith(("http://", "https://")):
            raise ValueError("base_url must start with http:// or https://")
        return clean.rstrip("/")

    @property
    def api_key_present(self) -> bool:
        return bool(self.api_key_env and os.getenv(self.api_key_env, "").strip())

    def get_api_key(self) -> str:
        """Return the secret value from the configured environment variable."""
        if not self.api_key_env:
            return ""
        return os.getenv(self.api_key_env, "").strip()


class LLMConfig(BaseModel):
    """LLM provider selection and profiles."""

    model_config = ConfigDict(extra="forbid")

    active_provider: str = "mock"
    profiles: dict[str, LLMProviderConfig]

    @model_validator(mode="after")
    def validate_active_profile(self) -> "LLMConfig":
        if self.active_provider not in self.profiles:
            raise ValueError(f"active_provider '{self.active_provider}' is not defined in llm.profiles")
        return self


class CameraConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: Literal["browser", "mock", "picamera2", "disabled"] = "browser"
    device_index: int = Field(default=0, ge=0)
    width: int = Field(default=640, gt=0)
    height: int = Field(default=480, gt=0)
    preview_fps: int = Field(default=12, gt=0, le=30)
    analysis_fps: int = Field(default=5, gt=0, le=15, validation_alias=AliasChoices("analysis_fps", "inference_fps"))
    rotation_degrees: Literal[0, 90, 180, 270] = 0
    flip_horizontal: bool = False
    flip_vertical: bool = False
    jpeg_quality: int = Field(default=75, ge=20, le=95)
    warmup_timeout_seconds: float = Field(default=5.0, gt=0)
    browser_fallback: bool = True

    @property
    def inference_fps(self) -> int:
        """Backward-compatible alias for older Phase 1 code."""
        return self.analysis_fps


class AudioConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_provider: Literal["browser", "bluetooth", "mock", "local", "disabled"] = "browser"
    output_provider: Literal["browser", "bluetooth", "mock", "local", "piper", "disabled"] = "browser"
    input_device: str = ""
    output_device: str = ""
    sample_rate: int = Field(default=16000, gt=0)
    channels: int = Field(default=1, gt=0, le=2)
    chunk_duration_ms: int = Field(default=30, gt=0, le=1000)
    buffer_max_chunks: int = Field(default=16, gt=0, le=256)
    max_recording_seconds: float = Field(default=8.0, gt=0, le=60)
    min_recording_ms: int = Field(default=250, ge=0, le=5000)
    half_duplex: bool = True
    barge_in_enabled: bool = False
    retain_temporary_audio: bool = False


class WakeWordConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: Literal["mock", "porcupine", "openwakeword", "disabled"] = "mock"
    model_path: str = ""
    phrase: str = "hey tutor"
    sensitivity: float = Field(default=0.5, ge=0.0, le=1.0)
    cooldown_seconds: float = Field(default=2.0, ge=0.0, le=60.0)
    activation_timeout_seconds: float = Field(default=5.0, gt=0.0, le=60.0)

    @model_validator(mode="after")
    def validate_model_path(self) -> "WakeWordConfig":
        if self.provider not in {"mock", "disabled"} and not self.model_path.strip():
            raise ValueError("wake_word.model_path is required for non-mock wake-word providers")
        return self


class VADConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: Literal["mock", "energy"] = "mock"
    speech_threshold: float = Field(default=0.01, ge=0.0, le=1.0)
    silence_timeout_ms: int = Field(default=800, gt=0, le=10000)
    max_recording_seconds: float = Field(default=8.0, gt=0.0, le=60.0)
    min_recording_ms: int = Field(default=250, ge=0, le=5000)
    noise_floor: float = Field(default=0.005, ge=0.0, le=1.0)


class STTRuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: Literal["mock", "legacy", "whisper_cpp", "disabled"] = "mock"
    model_path: str = ""
    executable_path: str = ""
    language: Literal["en", "hi"] = "en"
    timeout_seconds: float = Field(default=20.0, gt=0.0, le=180.0)
    mock_transcript: str = "A fraction is part of a whole."

    @model_validator(mode="after")
    def validate_local_provider(self) -> "STTRuntimeConfig":
        if self.provider == "whisper_cpp" and (not self.model_path.strip() or not self.executable_path.strip()):
            raise ValueError("stt.model_path and stt.executable_path are required for whisper_cpp")
        return self


class TTSRuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: Literal["mock", "legacy", "piper", "disabled"] = "mock"
    model_path: str = ""
    executable_path: str = ""
    voice: str = "default"
    language: Literal["en", "hi"] = "en"
    speed: float = Field(default=1.0, gt=0.25, le=4.0)
    volume: float = Field(default=1.0, ge=0.0, le=2.0)
    timeout_seconds: float = Field(default=20.0, gt=0.0, le=180.0)

    @model_validator(mode="after")
    def validate_piper(self) -> "TTSRuntimeConfig":
        if self.provider == "piper" and (not self.model_path.strip() or not self.executable_path.strip()):
            raise ValueError("tts.model_path and tts.executable_path are required for piper")
        return self


class RecognitionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    face_detection_provider: Literal["mock", "opencv"] = "mock"
    face_recognition_provider: Literal["mock", "local"] = "mock"
    face_detection_model_path: str = ""
    face_recognition_model_path: str = ""
    embedding_model_path: str = ""
    face_match_threshold: float = Field(default=0.72, gt=0.0, lt=1.0)
    unknown_label: str = "Guest"
    registration_sample_count: int = Field(default=3, ge=3, le=5)
    blur_threshold: float = Field(default=20.0, ge=0.0)
    brightness_min: float = Field(default=20.0, ge=0.0, le=255.0)
    brightness_max: float = Field(default=240.0, ge=0.0, le=255.0)
    recognition_interval_seconds: float = Field(default=1.0, gt=0.0)
    retain_temporary_media: bool = False

    @model_validator(mode="after")
    def validate_brightness_range(self) -> "RecognitionConfig":
        if self.brightness_min >= self.brightness_max:
            raise ValueError("recognition.brightness_min must be lower than brightness_max")
        if self.face_detection_provider == "opencv" and not self.face_detection_model_path.strip():
            raise ValueError("recognition.face_detection_model_path is required for OpenCV detection")
        if self.face_recognition_provider == "local" and not (
            self.face_recognition_model_path.strip() or self.embedding_model_path.strip()
        ):
            raise ValueError("recognition.face_recognition_model_path is required for the local recognizer")
        return self


class DatabaseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sqlite_path: str = "memory/app.db"


class AttentionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observation_window_seconds: float = Field(default=8.0, gt=0)
    confidence_threshold: float = Field(default=0.65, gt=0.0, lt=1.0)
    cooldown_seconds: float = Field(default=30.0, ge=0)


class EngagementConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    provider: Literal["mock", "local", "disabled"] = "mock"
    model_path: str = ""
    analysis_fps: int = Field(default=2, gt=0, le=10)
    rolling_window_seconds: float = Field(default=10.0, gt=0.0, le=120.0)
    absence_duration_seconds: float = Field(default=8.0, gt=0.0, le=120.0)
    response_inactivity_seconds: float = Field(default=20.0, gt=0.0, le=600.0)
    intervention_cooldown_seconds: float = Field(default=15.0, ge=0.0, le=600.0)
    max_interventions_per_lesson: int = Field(default=4, ge=0, le=20)
    abnormal_sample_ratio: float = Field(default=0.6, gt=0.0, le=1.0)
    minimum_samples: int = Field(default=3, ge=1, le=50)
    event_retention: int = Field(default=100, gt=0, le=1000)
    retain_raw_frames: bool = False

    @model_validator(mode="after")
    def validate_provider(self) -> "EngagementConfig":
        if self.retain_raw_frames:
            raise ValueError("engagement.retain_raw_frames must remain false")
        if self.provider == "local" and not self.model_path.strip():
            raise ValueError("engagement.model_path is required for the local provider")
        return self


class ServoConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    serial_port: str = ""
    pan_min_deg: float = -45.0
    pan_max_deg: float = 45.0
    tilt_min_deg: float = -20.0
    tilt_max_deg: float = 30.0
    deadband: float = Field(default=0.08, ge=0)
    rate_limit_hz: float = Field(default=5.0, gt=0)

    @model_validator(mode="after")
    def validate_limits(self) -> "ServoConfig":
        if self.pan_min_deg < -45.0 or self.pan_max_deg > 45.0 or self.pan_min_deg >= self.pan_max_deg:
            raise ValueError("servo pan limits must stay within -45..45 degrees")
        if self.tilt_min_deg < -20.0 or self.tilt_max_deg > 30.0 or self.tilt_min_deg >= self.tilt_max_deg:
            raise ValueError("servo tilt limits must stay within -20..30 degrees")
        return self


class ServoLimitsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_angle_deg: float = Field(default=-20.0, ge=-180.0, le=180.0)
    max_angle_deg: float = Field(default=20.0, ge=-180.0, le=180.0)
    max_speed_deg_per_second: float = Field(default=30.0, gt=0.0, le=360.0)
    max_duration_seconds: float = Field(default=1.0, gt=0.0, le=10.0)

    @model_validator(mode="after")
    def validate_angle_range(self) -> "ServoLimitsConfig":
        if self.min_angle_deg >= self.max_angle_deg:
            raise ValueError("hardware.servo_limits min_angle_deg must be lower than max_angle_deg")
        return self


class MotorLimitsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_speed_percent: float = Field(default=0.0, ge=0.0, le=100.0)
    max_duration_seconds: float = Field(default=0.0, ge=0.0, le=10.0)


class HardwareConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    provider: Literal["mock", "esp32", "disabled"] = "mock"
    transport: Literal["mock", "serial"] = "mock"
    physical_output_enabled: bool = False
    hardware_profile_approved: bool = False
    esp32_board: str = ""
    communication_method: str = ""
    serial_port: str = ""
    baud_rate: int = Field(default=115200, gt=0, le=1000000)
    power_supply: str = ""
    driver_board: str = ""
    emergency_stop_method: str = ""
    command_timeout_seconds: float = Field(default=1.0, gt=0.0, le=30.0)
    heartbeat_interval_seconds: float = Field(default=2.0, gt=0.0, le=60.0)
    heartbeat_timeout_seconds: float = Field(default=5.0, gt=0.0, le=120.0)
    retry_limit: int = Field(default=1, ge=0, le=5)
    stale_command_seconds: float = Field(default=5.0, gt=0.0, le=60.0)
    duplicate_retention: int = Field(default=100, gt=0, le=1000)
    audit_history_limit: int = Field(default=100, gt=0, le=1000)
    motion_cooldown_seconds: float = Field(default=0.5, ge=0.0, le=60.0)
    max_continuous_motion_seconds: float = Field(default=1.0, gt=0.0, le=10.0)
    unsafe_operating_zone_clear: bool = False
    emergency_stop_on_lost_connection: bool = True
    neutral_on_startup: bool = True
    neutral_on_shutdown: bool = True
    permitted_actions: list[Literal["neutral", "small_nod", "small_head_turn", "reset_position"]] = Field(
        default_factory=lambda: ["neutral", "small_nod", "small_head_turn", "reset_position"]
    )
    servo_limits: ServoLimitsConfig = Field(default_factory=ServoLimitsConfig)
    motor_limits: MotorLimitsConfig = Field(default_factory=MotorLimitsConfig)

    @model_validator(mode="after")
    def validate_physical_profile(self) -> "HardwareConfig":
        if self.physical_output_enabled:
            missing = [
                name
                for name, value in {
                    "esp32_board": self.esp32_board,
                    "communication_method": self.communication_method,
                    "serial_port": self.serial_port,
                    "power_supply": self.power_supply,
                    "driver_board": self.driver_board,
                    "emergency_stop_method": self.emergency_stop_method,
                }.items()
                if not str(value).strip()
            ]
            if self.provider != "esp32" or self.transport != "serial":
                raise ValueError("physical hardware output requires provider=esp32 and transport=serial")
            if not self.hardware_profile_approved or missing:
                raise ValueError(f"physical hardware output requires approved hardware profile; missing: {', '.join(missing)}")
        return self


class SensorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: Literal["mock", "dht", "auto"] = "mock"


class FeatureFlags(BaseModel):
    model_config = ConfigDict(extra="forbid")

    browser_camera: bool = True
    backend_camera_stream: bool = False
    student_registration: bool = False
    wake_word: bool = False
    servo_motion: bool = False
    student_ui: bool = False
    text_input: bool = True
    stt_input: bool = False
    tts_output: bool = False
    face_recognition: bool = False
    push_to_talk: bool = True
    voice_turns: bool = False
    barge_in: bool = False
    engagement_analysis: bool = False
    hardware_control: bool = False


class TeachingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    language: str = "en"
    level: str = "middle_school"
    prompt_version: str = "teaching_v1"
    max_remediation_attempts: int = Field(default=2, ge=0, le=5)
    max_lesson_turns: int = Field(default=8, gt=0, le=30)
    provider_timeout_seconds: float = Field(default=20.0, gt=0)
    structured_output_retries: int = Field(default=1, ge=0, le=3)
    session_inactivity_timeout_seconds: float = Field(default=900.0, gt=0)


class RuntimeConfig(BaseModel):
    """Complete application runtime configuration."""

    model_config = ConfigDict(extra="forbid")

    environment: str = "development"
    bind_host: str = "127.0.0.1"
    bind_port: int = Field(default=8000, gt=0, le=65535)
    llm: LLMConfig
    camera: CameraConfig = Field(default_factory=CameraConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    wake_word: WakeWordConfig = Field(default_factory=WakeWordConfig)
    vad: VADConfig = Field(default_factory=VADConfig)
    stt: STTRuntimeConfig = Field(default_factory=STTRuntimeConfig)
    tts: TTSRuntimeConfig = Field(default_factory=TTSRuntimeConfig)
    recognition: RecognitionConfig = Field(default_factory=RecognitionConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    attention: AttentionConfig = Field(default_factory=AttentionConfig)
    engagement: EngagementConfig = Field(default_factory=EngagementConfig)
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    servo: ServoConfig = Field(default_factory=ServoConfig)
    sensors: SensorConfig = Field(default_factory=SensorConfig)
    teaching: TeachingConfig = Field(default_factory=TeachingConfig)
    feature_flags: FeatureFlags = Field(default_factory=FeatureFlags)

    def validate_runtime(self) -> list[str]:
        """Return non-fatal startup warnings for missing optional secrets."""
        warnings: list[str] = []
        for name, profile in self.llm.profiles.items():
            if profile.provider != "mock" and profile.api_key_env and not profile.api_key_present:
                warnings.append(f"llm profile '{name}' is missing env var {profile.api_key_env}")
        if self.camera.provider == "picamera2":
            workers = os.getenv("WEB_CONCURRENCY", os.getenv("UVICORN_WORKERS", "1")).strip()
            if workers not in {"", "1"}:
                warnings.append("Picamera2 requires exactly one Uvicorn worker")
        if self.servo.enabled:
            warnings.append("Servo motion is enabled in config but Phase 1 only provides a mock controller")
        if self.bind_host not in {"127.0.0.1", "localhost", "::1"} and not os.getenv("ADMIN_API_TOKEN", "").strip():
            raise ConfigurationError("ADMIN_API_TOKEN is required when bind_host exposes the server beyond localhost")
        return warnings


def load_runtime_config(
    default_path: Path | str = DEFAULT_CONFIG_PATH,
    device_path: Path | str = DEVICE_CONFIG_PATH,
) -> RuntimeConfig:
    """Load, merge, and validate runtime configuration."""
    default_file = Path(default_path)
    if not default_file.exists():
        raise ConfigurationError(f"Default configuration file not found: {default_file}")

    raw = _read_yaml(default_file)
    device_file = Path(device_path)
    if device_file.exists():
        raw = _deep_merge(raw, _read_yaml(device_file))

    try:
        return RuntimeConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigurationError(str(exc)) from exc


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ConfigurationError(f"Configuration root must be an object: {path}")
    return loaded


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
