"""Application runtime factory and explicit provider registries."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from server.config import RuntimeConfig, load_runtime_config
from server.audio_pipeline import AudioRuntime
from server.engagement import EngagementRuntime
from server.hardware import HardwareRuntime, build_hardware_runtime
from server.interfaces import (
    AudioInputDriver,
    AudioOutputDriver,
    CameraDriver,
    FaceDetector,
    FaceRecognizer,
    MotionController,
    SensorProvider,
    SessionRepository,
    StudentRepository,
    WakeWordEngine,
)
from server.llm.providers import LLMProviderRegistry, build_llm_registry
from server.camera_pipeline import FramePipeline, build_camera_driver
from server.face_registration import FaceRegistrationService
from server.student_store import SQLiteStudentRepository
from server.teaching import TeachingOrchestrator
from server.mock_drivers import (
    MockAudioInputDriver,
    MockAudioOutputDriver,
    MockFaceDetector,
    MockFaceRecognizer,
    MockMotionController,
    MockSensorProvider,
    MockWakeWordEngine,
)


class RuntimeErrorConfig(Exception):
    """Raised when runtime construction fails."""


@dataclass(slots=True)
class DriverRegistry:
    camera: CameraDriver
    audio_input: AudioInputDriver
    audio_output: AudioOutputDriver
    wake_word: WakeWordEngine
    face_detector: FaceDetector
    face_recognizer: FaceRecognizer
    sensors: SensorProvider
    motion: MotionController
    students: StudentRepository
    sessions: SessionRepository


@dataclass(slots=True)
class ApplicationRuntime:
    config: RuntimeConfig
    llms: LLMProviderRegistry
    drivers: DriverRegistry
    camera_pipeline: FramePipeline
    audio: AudioRuntime
    engagement: EngagementRuntime
    hardware: HardwareRuntime
    teaching: TeachingOrchestrator
    student_store: SQLiteStudentRepository
    registration: FaceRegistrationService
    startup_warnings: list[str] = field(default_factory=list)

    def health(self) -> dict[str, Any]:
        """Return useful runtime metadata without exposing secrets."""
        active = self.config.llm.active_provider
        profile = self.config.llm.profiles[active]
        return {
            "environment": self.config.environment,
            "config_loaded": True,
            "active_llm_provider": active,
            "active_llm_model": profile.model,
            "active_llm_base_url": profile.base_url,
            "active_llm_credentials_configured": profile.provider == "mock" or profile.api_key_present,
            "registered_llm_providers": self.llms.names(),
            "camera_provider": self.config.camera.provider,
            "camera_state": self.camera_pipeline.status().state,
            "camera_resolution": f"{self.config.camera.width}x{self.config.camera.height}",
            "camera_sequence": self.camera_pipeline.status().sequence,
            "audio_input_provider": self.config.audio.input_provider,
            "audio_output_provider": self.config.audio.output_provider,
            "audio_state": self.audio.status().state,
            "wake_word_provider": self.config.wake_word.provider,
            "voice": {
                "audio": self.audio.status().safe_dict(),
                "push_to_talk_enabled": self.config.feature_flags.push_to_talk,
                "voice_turns_enabled": self.config.feature_flags.voice_turns,
                "barge_in_enabled": self.config.audio.barge_in_enabled and self.config.feature_flags.barge_in,
            },
            "sensor_provider": self.config.sensors.provider,
            "motion_provider": "hardware_runtime",
            "hardware": self.hardware.health(),
            "feature_flags": self.config.feature_flags.model_dump(),
            "database": {"sqlite_configured": bool(self.config.database.sqlite_path)},
            "recognition": {
                "face_detection_provider": self.config.recognition.face_detection_provider,
                "face_recognition_provider": self.config.recognition.face_recognition_provider,
                "registration_sample_count": self.config.recognition.registration_sample_count,
                "recognition_enabled": self.config.feature_flags.face_recognition,
                "provider_health": self.registration.health(),
            },
            "engagement": self.engagement.health(),
            "startup_warnings": self.startup_warnings,
        }


def build_application_runtime(config: RuntimeConfig | None = None) -> ApplicationRuntime:
    runtime_config = config or load_runtime_config()
    warnings = runtime_config.validate_runtime()
    llms = build_llm_registry(runtime_config.llm.profiles)
    camera_driver = build_camera_driver(runtime_config.camera)
    camera_pipeline = FramePipeline(camera_driver, runtime_config.camera)
    student_store = SQLiteStudentRepository(runtime_config.database.sqlite_path)
    registration = FaceRegistrationService(student_store, runtime_config.recognition)
    drivers = DriverRegistry(
        camera=camera_driver,
        audio_input=MockAudioInputDriver(),
        audio_output=MockAudioOutputDriver(),
        wake_word=MockWakeWordEngine(),
        face_detector=MockFaceDetector(),
        face_recognizer=MockFaceRecognizer(),
        sensors=MockSensorProvider(),
        motion=MockMotionController(runtime_config.servo),
        students=student_store,
        sessions=student_store,
    )
    teaching = TeachingOrchestrator(
        repository=drivers.sessions,
        llm_provider=llms.get(runtime_config.llm.active_provider),
        config=runtime_config.teaching,
    )
    audio = AudioRuntime(
        audio_config=runtime_config.audio,
        wake_config=runtime_config.wake_word,
        vad_config=runtime_config.vad,
        stt_config=runtime_config.stt,
        tts_config=runtime_config.tts,
        teaching=teaching,
    )
    engagement = EngagementRuntime(
        config=runtime_config.engagement,
        teaching=teaching,
        audio_status=audio.status,
    )
    hardware = build_hardware_runtime(runtime_config.hardware)
    return ApplicationRuntime(
        config=runtime_config,
        llms=llms,
        drivers=drivers,
        camera_pipeline=camera_pipeline,
        audio=audio,
        engagement=engagement,
        hardware=hardware,
        teaching=teaching,
        student_store=student_store,
        registration=registration,
        startup_warnings=warnings,
    )
