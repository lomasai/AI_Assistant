from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# This is the only file in the repository where a default value may be written.
# Every other module reads its numbers from a Config instance.

Strict = ConfigDict(extra="forbid")


class RuntimeConfig(BaseModel):
    model_config = Strict

    mode: Literal["debug", "user"] = "user"
    locale: str = "en"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    sinks: list[Literal["console", "jsonl"]] = Field(default_factory=lambda: ["console"])
    log_dir: str = "data/logs"
    event_replay_size: int = Field(default=512, ge=1)
    raise_on_handler_error: bool = True


class TenancyConfig(BaseModel):
    model_config = Strict

    org_id: str = "lomas-demo"
    school_id: str = "sunrise-nashik"
    class_id: str = "grade-6b"
    scratch_org_id: str = "scratch"


class StorageConfig(BaseModel):
    model_config = Strict

    backend: Literal["sqlite", "memory"] = "sqlite"
    path: str = "data/lomas.db"
    retention_days: int = Field(default=180, ge=1)
    purge_on_term_end: bool = True
    busy_timeout_ms: int = Field(default=5000, ge=0)


class SourceConfig(BaseModel):
    """One camera. `sources` is a list from the first commit, even with a
    single entry, so classroom CCTV and multi-angle capture arrive as config
    rather than as a redesign."""

    model_config = Strict

    id: str
    kind: Literal["picamera2", "usb", "rtsp", "file", "folder", "mock"] = "mock"
    zone: str = "front"
    enabled: bool = True
    width: int = Field(default=1280, ge=1)
    height: int = Field(default=720, ge=1)
    fps: int = Field(default=15, ge=0)  # 0 means capture as fast as the device allows
    zoom: float = Field(default=1.0, ge=1.0, le=2.5)
    rotation: Literal[0, 90, 180, 270] = 0
    device: str | int | None = None  # usb index or device path
    path: str | None = None  # file or folder source
    url: str | None = None  # rtsp source
    loop: bool = True  # replay sources start over at the end


class VisionConfig(BaseModel):
    model_config = Strict

    buffer_size: int = Field(default=4, ge=1)
    read_timeout_ms: int = Field(default=200, ge=1)


class PoseConfig(BaseModel):
    """Calibration for the geometric head-pose approximation. These are the
    knobs to turn if the attention cone feels wrong in a real classroom."""

    model_config = Strict

    yaw_scale_degrees: float = Field(default=60.0, gt=0)
    pitch_scale_degrees: float = Field(default=60.0, gt=0)
    pitch_neutral: float = Field(default=0.5, ge=0.0, le=1.0)


class FaceConfig(BaseModel):
    model_config = Strict

    enabled: bool = True
    detector: Literal["yunet", "mediapipe", "mock"] = "yunet"
    model_path: str = "models/face_detection_yunet_2023mar.onnx"
    detect_fps: int = Field(default=10, ge=1)
    downscale_width: int = Field(default=640, ge=64)
    min_confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    nms_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=1)
    min_face_px: int = Field(default=40, ge=1)
    max_tracks: int = Field(default=8, ge=1)
    track_iou_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    track_birth_hits: int = Field(default=3, ge=1)
    track_death_seconds: float = Field(default=1.5, gt=0)
    pose: PoseConfig = Field(default_factory=PoseConfig)

    # Recognition. Identity is resolved on new tracks and then carried by the
    # tracker, so these govern how rarely the embedder runs.
    embedder: Literal["arcface_onnx", "mock"] = "arcface_onnx"
    embedder_model_path: str = "models/mobilefacenet.onnx"
    embedding_dim: int = Field(default=512, ge=1)
    match_threshold: float = Field(default=0.38, gt=0.0, le=2.0)
    reverify_seconds: float = Field(default=20.0, gt=0)
    unknown_after_attempts: int = Field(default=3, ge=1)
    recognition_min_face_px: int = Field(default=80, ge=1)
    crop_margin: float = Field(default=0.2, ge=0.0, le=1.0)


class EnrolmentConfig(BaseModel):
    model_config = Strict

    sweep_seconds: float = Field(default=4.0, gt=0)
    sample_frames: int = Field(default=15, ge=1)
    keep_best: int = Field(default=3, ge=1)
    min_quality: float = Field(default=0.5, ge=0.0, le=1.0)
    min_face_px: int = Field(default=96, ge=1)
    required_angles: list[str] = Field(default_factory=lambda: ["left", "centre", "right"])
    angle_yaw_degrees: float = Field(default=20.0, gt=0)
    sharpness_reference: float = Field(default=120.0, gt=0)
    crop_margin: float = Field(default=0.2, ge=0.0, le=1.0)


class PrivacyConfig(BaseModel):
    model_config = Strict

    # A school that will not consent to face recognition still gets a working
    # teaching assistant; it simply addresses the room rather than a child.
    recognition_enabled: bool = True
    require_consent: bool = True

    # Not a bool on purpose. There is no code path that stores an image and no
    # column to put one in, so the config must not be able to promise one.
    store_images: Literal[False] = False

    store_attention_detail: bool = False


class AttentionConfig(BaseModel):
    model_config = Strict

    enabled: bool = True
    cone_yaw_degrees: float = Field(default=35.0, gt=0)
    cone_pitch_degrees: float = Field(default=25.0, gt=0)
    window_seconds: float = Field(default=10.0, gt=0)
    threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    min_duration_seconds: float = Field(default=6.0, gt=0)
    cooldown_seconds: float = Field(default=120.0, ge=0)
    max_nudges_per_session: int = Field(default=3, ge=0)


class WakeConfig(BaseModel):
    model_config = Strict

    enabled: bool = True
    engine: Literal["openwakeword", "porcupine", "keyboard"] = "openwakeword"
    phrase: str = "hey lomas"
    sensitivity: float = Field(default=0.6, ge=0.0, le=1.0)
    model_path: str = "models/wake/hey_lomas.onnx"
    zone: str = "front"
    access_key_env: str = "PICOVOICE_ACCESS_KEY"  # porcupine only


class SttConfig(BaseModel):
    model_config = Strict

    engine: Literal["groq", "vosk", "keyboard"] = "groq"
    # Used when the internet is gone, which in a government school is often.
    fallback_engine: Literal["groq", "vosk", "keyboard"] = "vosk"
    language: str = "en"
    model: str = "whisper-large-v3-turbo"
    model_path: str = "models/vosk"  # vosk keeps one directory per language
    api_base: str = "https://api.groq.com/openai/v1"
    api_key_env: str = "GROQ_API_KEY"  # the name is config, the secret is not
    sample_rate: int = Field(default=16000, ge=8000)
    silence_timeout_ms: int = Field(default=1200, ge=1)
    max_utterance_seconds: int = Field(default=20, ge=1)
    timeout_seconds: float = Field(default=15.0, gt=0)


def _default_voices() -> dict[str, str]:
    return {"en": "en_US-lessac-medium", "hi": "hi_IN-pratham-medium"}


class TtsConfig(BaseModel):
    model_config = Strict

    engine: Literal["piper", "gtts", "null"] = "piper"
    rate: float = Field(default=1.0, gt=0)
    # A map per language, never a single voice string - that is what keeps
    # adding a language content work rather than code work.
    voice: dict[str, str] = Field(default_factory=_default_voices)
    fallback_language: str = "en"
    binary: str = "piper"
    model_dir: str = "models/piper"
    scratch_file: str = "data/tts-out.mp3"


class AudioInputConfig(BaseModel):
    model_config = Strict

    id: str = "main"
    device: str = "default"
    zone: str = "front"


def _default_inputs() -> list[AudioInputConfig]:
    return [AudioInputConfig()]


class AudioConfig(BaseModel):
    model_config = Strict

    # A list from the first commit. Per-desk microphones become entries here.
    inputs: list[AudioInputConfig] = Field(default_factory=_default_inputs)
    output: str = "default"
    half_duplex: bool = True
    tail_ms: int = Field(default=250, ge=0)


class SpeechConfig(BaseModel):
    model_config = Strict

    wake: WakeConfig = Field(default_factory=WakeConfig)
    stt: SttConfig = Field(default_factory=SttConfig)
    tts: TtsConfig = Field(default_factory=TtsConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)


def _default_sources() -> list[SourceConfig]:
    return [SourceConfig(id="head")]


class Config(BaseModel):
    model_config = Strict

    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    tenancy: TenancyConfig = Field(default_factory=TenancyConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    sources: list[SourceConfig] = Field(default_factory=_default_sources)
    vision: VisionConfig = Field(default_factory=VisionConfig)
    face: FaceConfig = Field(default_factory=FaceConfig)
    attention: AttentionConfig = Field(default_factory=AttentionConfig)
    enrolment: EnrolmentConfig = Field(default_factory=EnrolmentConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)
    speech: SpeechConfig = Field(default_factory=SpeechConfig)

    @property
    def is_debug(self) -> bool:
        return self.runtime.mode == "debug"

    @property
    def active_org_id(self) -> str:
        """Debug runs write to a scratch tenant so bench testing never lands
        in a real school's data."""
        return self.tenancy.scratch_org_id if self.is_debug else self.tenancy.org_id
