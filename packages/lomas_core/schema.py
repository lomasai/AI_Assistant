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

    # Vision publishes on every detect cycle. Writing all of that to the
    # session log would bury the events a report actually reads.
    log_event_exclude: list[str] = Field(default_factory=lambda: ["vision.*"])


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


class PipelineConfig(BaseModel):
    """The join between frames and faces. It lives in the app, but its knobs
    belong here with everything else that can be turned."""

    model_config = Strict

    enabled: bool = True
    # Which camera feeds recognition. Empty means the first enabled source,
    # which is what a one-camera robot wants and a CCTV room overrides.
    source: str = ""
    publish_tracks: bool = True
    idle_sleep_seconds: float = Field(default=0.01, gt=0)
    join_timeout_seconds: float = Field(default=2.0, gt=0)

    # One failed cycle is a dropped frame. Every cycle failing is a missing
    # model or an unplugged camera, and retrying that ten times a second for
    # an hour helps nobody.
    max_consecutive_errors: int = Field(default=5, ge=1)


class VisionConfig(BaseModel):
    model_config = Strict

    buffer_size: int = Field(default=4, ge=1)
    read_timeout_ms: int = Field(default=200, ge=1)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)


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

    # The consent row that has to exist before a single vector is written.
    consent_kind: str = "face_recognition"
    # An enrolment left half finished must not hold a child's vectors in
    # memory until the robot is switched off.
    abandoned_after_seconds: float = Field(default=120.0, gt=0)


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

    # Where the finished audio goes. Synthesising it and dropping it on the
    # floor is the difference between a robot that teaches and one that mimes.
    player: str = "auto"  # auto | none | winsound | aplay | afplay | ffplay | mpg123
    player_command: str = ""  # an exact command line, when auto guesses wrong
    sample_rate: int = Field(default=22050, ge=8000)  # piper voices are 22.05 kHz


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


class EndpointConfig(BaseModel):
    model_config = Strict

    api_base: str = ""
    api_key_env: str = ""
    model: str = ""
    # Anthropic only. `effort` replaces temperature on Opus 5, which rejects
    # sampling parameters outright.
    effort: Literal["low", "medium", "high", "xhigh", "max"] = "low"
    server_side_fallback: bool = True


def _default_endpoints() -> dict[str, EndpointConfig]:
    return {
        "groq": EndpointConfig(
            api_base="https://api.groq.com/openai/v1",
            api_key_env="GROQ_API_KEY",
            model="llama-3.3-70b-versatile",
        ),
        "anthropic": EndpointConfig(
            api_base="https://api.anthropic.com/v1",
            api_key_env="ANTHROPIC_API_KEY",
            model="claude-opus-5",
        ),
        "openai": EndpointConfig(
            api_base="https://api.openai.com/v1",
            api_key_env="OPENAI_API_KEY",
            model="gpt-4o-mini",
        ),
    }


class RouterConfig(BaseModel):
    """Score a question, pick one model. Never chain them - that multiplies
    latency and cost for very little gain."""

    model_config = Strict

    enabled: bool = True
    long_sentence_words: int = Field(default=25, ge=1)
    reasoning_keywords: list[str] = Field(
        default_factory=lambda: ["why", "how", "explain", "compare", "prove", "derive"]
    )
    weight_length: int = Field(default=1, ge=0)
    weight_keywords: int = Field(default=1, ge=0)
    weight_multipart: int = Field(default=1, ge=0)
    medium_at: int = Field(default=2, ge=1)
    complex_at: int = Field(default=3, ge=1)
    simple_provider: str = "groq"
    medium_provider: str = "groq"
    complex_provider: str = "anthropic"


class LlmConfig(BaseModel):
    model_config = Strict

    provider: Literal["offline", "groq", "anthropic", "openai"] = "offline"
    model: str = ""  # empty means the endpoint's own default
    # Used on timeout, rate limit, or no internet at all.
    fallback_provider: Literal["offline", "groq", "anthropic", "openai"] = "offline"
    temperature: float = Field(default=0.4, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, ge=1)
    timeout_seconds: float = Field(default=12.0, gt=0)
    retries: int = Field(default=2, ge=0)
    stream: bool = True
    prompts_path: str = "config/prompts"
    fallback_language: str = "en"
    offline_faq: str = "offline"
    endpoints: dict[str, EndpointConfig] = Field(default_factory=_default_endpoints)
    router: RouterConfig = Field(default_factory=RouterConfig)


def _default_sequence() -> list[str]:
    return ["attendance", "greeting", "lesson", "interaction", "quiz", "wrapup"]


def _default_timeouts() -> dict[str, float]:
    return {
        "attendance": 120.0,
        "greeting": 30.0,
        "lesson": 900.0,
        "interaction": 420.0,
        "quiz": 300.0,
        "wrapup": 90.0,
    }


class FlowConfig(BaseModel):
    model_config = Strict

    # A robot boots, shows a sleeping face and waits to be told to begin.
    # true teaches one class and exits, which is what a bench run wants.
    autostart: bool = True

    # Reordering or dropping a stage is an edit here, never a code change.
    sequence: list[str] = Field(default_factory=_default_sequence)
    stage_timeout_seconds: dict[str, float] = Field(default_factory=_default_timeouts)
    default_timeout_seconds: float = Field(default=300.0, gt=0)
    tick_seconds: float = Field(default=0.25, gt=0)
    pause_poll_seconds: float = Field(default=0.1, gt=0)

    questions_per_lesson: int = Field(default=6, ge=0)
    # How long to wait on a quiz question before moving on. A class where
    # nobody answers still has to reach the end of the lesson.
    answer_wait_seconds: float = Field(default=20.0, gt=0)
    quiz_length: int = Field(default=6, ge=0)
    pass_mark: float = Field(default=0.6, ge=0.0, le=1.0)

    # Until vision is wired in, and wherever recognition is switched off, the
    # class list stands in for recognised faces so the lesson still has names.
    # How long to wait for recognised faces before the class list stands in.
    # Without this the stage would sit out its whole timeout on a robot with
    # no camera running.
    attendance_wait_seconds: float = Field(default=15.0, ge=0)
    attendance_falls_back_to_roster: bool = True


class HardwareConfig(BaseModel):
    """The body. Off by default, because most of the time there isn't one.

    The Pi sends intent and the ESP32 does execution, so nothing here is a
    servo angle or a safety threshold - those live in config/hardware/ and
    are applied on the board.
    """

    model_config = Strict

    enabled: bool = False
    backend: Literal["simulator", "esp32"] = "simulator"

    port: str = "/dev/ttyUSB0"
    baud: int = Field(default=921600, ge=9600)
    timeout_seconds: float = Field(default=0.2, gt=0)
    telemetry_hz: int = Field(default=20, ge=1, le=255)

    config_path: str = "config/hardware"
    gesture_speed: int = Field(default=100, ge=1, le=100)

    # Debug mode logs the exact bytes, which is what gets compared against the
    # board's own serial log when something does not move.
    log_frames: bool = True
    simulate_travel_time: bool = True
    simulated_battery_mv: int = Field(default=12000, ge=0)

    # The head following whoever is speaking. Config, because a robot whose
    # head tracks children is not something to switch on without asking.
    look_at_enabled: bool = True
    # A deadband. A head that snaps between children every tenth of a second
    # is unsettling to watch and hard on the servos.
    look_at_min_degrees: float = Field(default=6.0, ge=0)

    # Event to gesture. Adding one is a line here, never a change in code.
    gestures: dict[str, str] = Field(
        default_factory=lambda: {
            "session.opened": "namaste",
            "session.closed": "namaste",
            "question.asked": "thinking",
            "quiz.marked": "celebrate",
        }
    )


class ScreenConfig(BaseModel):
    model_config = Strict

    enabled: bool = True
    output: str = ""  # xrandr name, for the second HDMI port
    width: int = Field(default=1024, ge=1)
    height: int = Field(default=600, ge=1)
    scale: float = Field(default=1.0, gt=0)


class DisplayConfig(BaseModel):
    """Two surfaces on one Pi. The 7-inch chest panel is the robot's face and
    cannot carry lesson text a class can read; the second HDMI port drives a
    classroom TV or projector for that."""

    model_config = Strict

    face_screen: ScreenConfig = Field(default_factory=ScreenConfig)
    board_screen: ScreenConfig = Field(
        default_factory=lambda: ScreenConfig(enabled=False, output="HDMI-2",
                                             width=1920, height=1080)
    )
    # Where text starts on the face panel. Absurd on a laptop, right at 1.5 m.
    base_font_px: int = Field(default=32, ge=1)


class WebConfig(BaseModel):
    model_config = Strict

    enabled: bool = True
    host: str = "0.0.0.0"
    port: int = Field(default=8080, ge=1, le=65535)
    surfaces: list[str] = Field(default_factory=lambda: ["face", "board", "teacher"])

    mjpeg_quality: int = Field(default=70, ge=1, le=100)
    mjpeg_fps: int = Field(default=15, ge=1)
    mjpeg_source: str = ""  # empty means the pipeline's camera

    # Per browser. When it is full the oldest event goes: a tab left open on
    # a locked laptop must never slow the lesson down.
    client_queue: int = Field(default=64, ge=1)
    ping_seconds: float = Field(default=20.0, gt=0)
    # Vision publishes on every detect cycle and the face UI needs it, so it
    # is on the wire even though it is kept out of the session log.
    event_filter: list[str] = Field(default_factory=lambda: ["*"])
    shutdown_seconds: float = Field(default=2.0, gt=0)


def _default_latencies() -> dict[str, list[str]]:
    """Event pairs to time. Adding one - a wake word to its transcript, once
    there is a microphone loop - is a line here, not a change in code."""
    return {
        "speak": ["robot.say", "robot.spoke"],
        "answer": ["question.asked", "question.answered"],
        "marking": ["quiz.recorded", "quiz.marked"],
    }


class DebugConfig(BaseModel):
    """The diagnostics overlay. Served in debug mode and nowhere else."""

    model_config = Strict

    enabled: bool = True
    rate_window_seconds: float = Field(default=5.0, gt=0)
    rate_samples: int = Field(default=200, ge=2)
    rate_events: list[str] = Field(
        default_factory=lambda: ["vision.tracks", "robot.say", "student.identified"]
    )
    latencies: dict[str, list[str]] = Field(default_factory=_default_latencies)
    keep_samples: int = Field(default=20, ge=1)
    keep_events: int = Field(default=200, ge=1)
    tracks_event: str = "vision.tracks"

    # Ten a second would push everything else out of the event list.
    noisy_events: list[str] = Field(default_factory=lambda: ["vision.tracks"])

    # Empty means unpriced, and the panel says so rather than inventing a
    # number. Keys are model ids; values are [input, output] per million.
    cost_per_million: dict[str, list[float]] = Field(default_factory=dict)
    currency: str = "USD"

    poll_seconds: float = Field(default=1.0, gt=0)


class TeacherConfig(BaseModel):
    """The surface that decides whether teachers keep using the product."""

    model_config = Strict

    enabled: bool = True
    recent_sessions: int = Field(default=10, ge=1)

    # Not a bool. There is no code path that puts an attention score in a
    # report and no key for one, so the config must not be able to promise it.
    report_shows_attention: Literal[False] = False


class AgentConfig(BaseModel):
    """One agent's overrides. Empty strings inherit from llm.*, so the safety
    filter can be pinned to a cheap fast model without touching the tutor."""

    model_config = Strict

    provider: str = ""
    model: str = ""
    prompt: str = ""  # the agent's main prompt file
    # Extra prompt files the agent uses, by role. Marking an answer and
    # writing a question are different jobs and want different instructions.
    prompts: dict[str, str] = Field(default_factory=dict)
    max_tokens: int = Field(default=0, ge=0)  # 0 inherits llm.max_tokens


def _default_agent_settings() -> dict[str, AgentConfig]:
    return {
        "tutor": AgentConfig(prompt="tutor"),
        "quizmaster": AgentConfig(prompt="quizmaster", prompts={"mark": "marking"}),
        "narrator": AgentConfig(prompt="narrator"),
        "engagement": AgentConfig(prompt="nudge"),
        "safety": AgentConfig(prompt="safety"),
    }


class SafetyConfig(BaseModel):
    model_config = Strict

    # The term list runs with no model and no internet, so it is the only
    # check a school can rely on being there. It is empty by default because
    # what a board wants blocked is their decision, not ours.
    blocked_terms: list[str] = Field(default_factory=list)

    # The model check costs a round trip on every line the robot speaks.
    use_model: bool = False
    allow_token: str = "ALLOW"
    block_token: str = "BLOCK"

    # A verdict that is neither word means the model is unusable for this -
    # the offline provider, a timeout, a refusal. A robot that falls silent
    # mid-lesson is a failed class; the term list still stands.
    fail_open: bool = True


class AgentsConfig(BaseModel):
    model_config = Strict

    # Removing a name here switches that agent off. Nothing else changes.
    enabled: list[str] = Field(
        default_factory=lambda: ["tutor", "quizmaster", "narrator", "engagement", "safety"]
    )
    settings: dict[str, AgentConfig] = Field(default_factory=_default_agent_settings)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)


class ContextConfig(BaseModel):
    """What an agent is allowed to see. Every number here narrows it."""

    model_config = Strict

    mcp_enabled: bool = True
    history_turns: int = Field(default=8, ge=0)
    lesson_window: int = Field(default=3, ge=0)  # taught segments, not the whole lesson
    include_student_profile: bool = True
    recent_answers: int = Field(default=5, ge=0)


class ContentConfig(BaseModel):
    model_config = Strict

    language: str = "en"
    grade: str = "6"
    subject: str = "science"
    vocabulary_level: Literal["primary", "middle", "secondary"] = "middle"
    pack_path: str = "content"
    default_topic: str = "photosynthesis"


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
    llm: LlmConfig = Field(default_factory=LlmConfig)
    flow: FlowConfig = Field(default_factory=FlowConfig)
    content: ContentConfig = Field(default_factory=ContentConfig)
    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    display: DisplayConfig = Field(default_factory=DisplayConfig)
    web: WebConfig = Field(default_factory=WebConfig)
    teacher: TeacherConfig = Field(default_factory=TeacherConfig)
    debug: DebugConfig = Field(default_factory=DebugConfig)
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)

    @property
    def is_debug(self) -> bool:
        return self.runtime.mode == "debug"

    @property
    def active_org_id(self) -> str:
        """Debug runs write to a scratch tenant so bench testing never lands
        in a real school's data."""
        return self.tenancy.scratch_org_id if self.is_debug else self.tenancy.org_id
