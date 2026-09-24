from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# This is the only file in the repository where a default value may be written.
# Every other module reads its numbers from a Config instance.

Strict = ConfigDict(extra="forbid")


class RuntimeConfig(BaseModel):
    model_config = Strict

    mode: Literal["debug", "user"] = "user"

    # Which profile produced this config. Set by the loader, not by hand:
    # pi and demo both resolve to a mode, and "which run was this" is not
    # answerable from the mode alone when comparing two traces.
    profile: str = ""
    locale: str = "en"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    sinks: list[Literal["console", "jsonl"]] = Field(default_factory=lambda: ["console"])
    log_dir: str = "data/logs"
    event_replay_size: int = Field(default=512, ge=1)

    # Derived from the mode unless it is set explicitly. A profile that says
    # `mode: user` and then ends a lesson because one subscriber threw is a
    # profile that lies, and profiles do not inherit from each other - pi.yaml
    # is built on default.yaml, not on user.yaml.
    raise_on_handler_error: bool | None = None

    # Vision publishes on every detect cycle. Writing all of that to the
    # session log would bury the events a report actually reads.
    log_event_exclude: list[str] = Field(default_factory=lambda: ["vision.*"])


    @model_validator(mode="after")
    def _error_policy_follows_the_mode(self) -> "RuntimeConfig":
        if self.raise_on_handler_error is None:
            # On the bench a broken subscriber should stop everything. In a
            # classroom it must not end the lesson in front of forty children.
            object.__setattr__(self, "raise_on_handler_error", self.mode == "debug")
        return self


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


class CameraControls(BaseModel):
    """Exposure and colour, per camera.

    A classroom in the afternoon and the same room at four o'clock are two
    different cameras, and the answer is a number here rather than a brighter
    bulb. Zero means "let the sensor decide" throughout.
    """

    model_config = Strict

    # Auto-exposure needs a second or two of real frames before it settles.
    # Grabbing immediately after start is why a good sensor produces a dark
    # picture, and it is the single most common camera complaint on a Pi.
    warmup_seconds: float = Field(default=2.5, ge=0.0)

    auto_exposure: bool = True
    auto_white_balance: bool = True

    # AE compensation in stops. The first thing to raise in a dim room: +1.0
    # is twice the light, and it costs nothing but a little noise.
    exposure_value: float = Field(default=0.0, ge=-8.0, le=8.0)

    # Both zero lets auto-exposure choose. Set them only to pin it.
    exposure_time_us: int = Field(default=0, ge=0)
    analogue_gain: float = Field(default=0.0, ge=0.0, le=16.0)

    brightness: float = Field(default=0.0, ge=-1.0, le=1.0)
    contrast: float = Field(default=1.0, ge=0.0, le=32.0)
    saturation: float = Field(default=1.0, ge=0.0, le=32.0)


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
    controls: CameraControls = Field(default_factory=CameraControls)


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

    # Movement around the mouth, which is how the robot tells which of two
    # known faces is the one talking. Cheap: a 24 by 24 patch per face per
    # cycle. See the `mouth_motion` speaker resolver.
    watch_mouths: bool = True
    mouth_smoothing: float = Field(default=0.6, ge=0.0, lt=1.0)
    mouth_patch_px: int = Field(default=24, ge=8)

    # Recognition. Identity is resolved on new tracks and then carried by the
    # tracker, so these govern how rarely the embedder runs.
    # sface by default: it runs on the OpenCV already installed for the
    # detector. arcface_onnx needs onnxruntime and a model with no one obvious
    # place to get it, which is why a Pi showed "recognition off" for weeks.
    # Straighten a face on the eyes before embedding it. What the model was
    # trained on, and worth a few percent; off falls back to a plain crop.
    align: bool = True
    embedder: Literal["sface", "arcface_onnx", "mock"] = "sface"
    embedder_model_path: str = "models/face_recognition_sface_2021dec.onnx"
    embedding_dim: int = Field(default=128, ge=1)
    # Cosine distance, lower is stricter. SFace's published same-person line is
    # a similarity of 0.363, a distance of 0.637; this sits a little inside it
    # because the crops are resized rather than aligned.
    match_threshold: float = Field(default=0.6, gt=0.0, le=2.0)
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

    # The robot enrolling a child itself: "I do not think we have met. What
    # is your name?" - then the sweep, coached out loud. Off by default,
    # because who may consent to a child's face being stored is a school's
    # decision and not a default. On, it is recorded as consented by the
    # robot with whoever set it up answerable for that.
    by_voice: bool = False
    voice_consent_by: str = "robot"
    stranger_after_seconds: float = Field(default=4.0, gt=0)
    # How long the robot waits for a name before letting it go.
    name_wait_seconds: float = Field(default=20.0, gt=0)
    frame_gap_seconds: float = Field(default=0.2, gt=0)
    name_lead_ins: list[str] = Field(
        default_factory=lambda: ["my name is", "i am", "i'm", "this is", "it is",
                                 "mera naam", "main"]
    )

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
    # A transcript this short is punctuation or a hallucination, not a child.
    min_characters: int = Field(default=3, ge=0)
    max_utterance_seconds: int = Field(default=20, ge=1)
    timeout_seconds: float = Field(default=15.0, gt=0)


def _default_voices() -> dict[str, str]:
    return {"en": "en_US-lessac-medium", "hi": "hi_IN-pratham-medium"}


class VolumeConfig(BaseModel):
    """How loud the robot is, and who is allowed to change it.

    A classroom is not a lab: the same robot is too quiet for forty children
    after lunch and too loud for six of them at the back of a library. The
    teacher needs a knob, and the knob has to survive a reboot, so this is
    config with a level that is written back to disk.
    """

    model_config = Strict

    # alsa moves the card's own mixer, which is the real knob and the one a
    # headphone jack obeys. software scales the samples before they reach the
    # player, which works on a laptop and on a card with no mixer at all.
    control: Literal["auto", "alsa", "software", "none"] = "auto"

    level: float = Field(default=0.8, ge=0.0, le=1.0)
    muted: bool = False

    # Loudness is not linear. Halfway along the slider should sound halfway,
    # and samples scaled by 0.5 sound much louder than that - so the software
    # control raises the level to this power before touching the audio.
    curve: float = Field(default=2.0, gt=0)

    # The slider's own limits. A school that never wants the robot above
    # conversation volume sets max_level and the teacher cannot go past it.
    min_level: float = Field(default=0.0, ge=0.0, le=1.0)
    max_level: float = Field(default=1.0, ge=0.0, le=1.0)
    step: float = Field(default=0.05, gt=0, le=1.0)

    # alsa only. Empty takes the card out of speech.tts.player_device, which
    # is where the speaker already is.
    card: str = ""
    # The first of these the card actually has wins. Different Pi audio
    # devices call the same knob different things.
    mixers: list[str] = Field(default_factory=lambda: [
        "PCM", "Headphone", "Speaker", "Master", "Digital", "Playback",
    ])
    mixer_timeout_seconds: float = Field(default=3.0, gt=0)

    # How many decibels the slider spans, from full down to its bottom end.
    #
    # A card's own percentage is no use as a level. The Pi's jack runs from
    # -102 dB to +4, so 80% of it is -17 dB - a seventh of the amplitude,
    # and inaudible under a fan. Halving the loudness is about -10 dB, so a
    # slider worth having covers the top thirty or forty of them and the
    # bottom of it is quiet rather than off.
    range_db: float = Field(default=40.0, gt=0)

    # Where the level set from the teacher's page is kept, so a robot that
    # is switched off at the wall comes back at the volume the room chose.
    remember: bool = True
    state_file: str = "data/volume.json"


class TtsConfig(BaseModel):
    model_config = Strict

    engine: Literal["piper", "piper_python", "gtts", "null"] = "piper"
    rate: float = Field(default=1.0, gt=0)
    # A map per language, never a single voice string - that is what keeps
    # adding a language content work rather than code work.
    voice: dict[str, str] = Field(default_factory=_default_voices)
    fallback_language: str = "en"

    # gtts only. Google serves a different accent per domain, and co.in is
    # Indian English - the one a class here will find familiar. piper has no
    # Indian English voice at all, which is why this exists.
    accent: str = "com"
    binary: str = "piper"
    model_dir: str = "models/piper"
    # piper_python only. Load the voice at boot rather than on the first
    # sentence, so the greeting is not the thing that waits.
    preload: bool = True
    scratch_file: str = "data/tts-out.mp3"

    # Where the finished audio goes. Synthesising it and dropping it on the
    # floor is the difference between a robot that teaches and one that mimes.
    player: str = "auto"  # auto | none | winsound | aplay | afplay | ffplay | mpg123
    # A Pi has four playback cards - the jack, two HDMI, often a DAC - and
    # ALSA's `default` is rarely the one with a speaker on it. `aplay -l`
    # gives the name; this is plughw:CARD=<name>,DEV=0. Not the number, which
    # moves between boots.
    player_device: str = ""
    player_command: str = ""  # an exact command line, when auto guesses wrong

    # The longest a lesson waits on one sentence before moving on. A cloud
    # voice that never answers must not stop the class.
    volume: VolumeConfig = Field(default_factory=VolumeConfig)

    utterance_timeout_seconds: float = Field(default=60.0, gt=0)
    stop_seconds: float = Field(default=2.0, gt=0)
    sample_rate: int = Field(default=22050, ge=8000)  # piper voices are 22.05 kHz


class AudioInputConfig(BaseModel):
    model_config = Strict

    id: str = "main"
    device: str = "default"
    zone: str = "front"


def _default_inputs() -> list[AudioInputConfig]:
    return [AudioInputConfig()]


class SpeakerConfig(BaseModel):
    """Who just spoke, and how hard the robot may work to find out.

    The chain is a list so an experiment is an edit: `[mouth_motion]` alone
    tests mouth movement, `[tapped]` is the teacher tapping a name and
    nothing else.
    """

    model_config = Strict

    resolvers: list[str] = Field(
        default_factory=lambda: ["tapped", "single_face", "recent", "spoken_name",
                                 "only_student", "ask"]
    )

    # How long a face counts as being in front of the robot after the camera
    # last saw it. Shorter than a blink is a robot that forgets mid-sentence.
    visible_seconds: float = Field(default=6.0, gt=0)

    # How long one speaker keeps the turn. This is what makes saying a name
    # bearable: once, then a conversation.
    recent_seconds: float = Field(default=90.0, ge=0)

    # 1.0 demands the spelling in the register. Speech to text returns
    # Akshaya, Akash and action for the same child, so it does not.
    name_match: float = Field(default=0.75, ge=0.0, le=1.0)
    # Asked for on a name with no lead-in before it, which is easier to get
    # wrong: "Meera" said by anyone, in a lesson about Meera's garden.
    no_cue_extra: float = Field(default=0.1, ge=0.0, le=1.0)
    name_window_words: int = Field(default=6, ge=1)

    # `mouth_motion`: how much movement counts as speaking at all, and how
    # far ahead of the next face it has to be before the robot will name
    # somebody on the strength of it.
    mouth_min_score: float = Field(default=0.012, ge=0.0)
    mouth_clear_by: float = Field(default=0.006, ge=0.0)
    name_cues: list[str] = Field(
        default_factory=lambda: ["i am", "i'm", "my name is", "this is", "here is",
                                 "मेरा नाम", "मैं"]
    )
    # Taken off the front of what is left, so the tutor is asked the question
    # and not the preamble.
    question_lead_ins: list[str] = Field(
        default_factory=lambda: ["my question is", "i want to ask", "i wanted to ask",
                                 "i have a question", "and", "so"]
    )
    strip_name: bool = True


class AudioConfig(BaseModel):
    model_config = Strict

    # A list from the first commit. Per-desk microphones become entries here.
    inputs: list[AudioInputConfig] = Field(default_factory=_default_inputs)
    output: str = "default"
    half_duplex: bool = True
    tail_ms: int = Field(default=250, ge=0)

    # Hearing. arecord is on every Raspberry Pi already, so the default costs
    # no install; sounddevice is for machines without it.
    recorder: str = "auto"  # auto | none | arecord | sounddevice
    recorder_command: str = ""
    device: str = ""  # ALSA name, e.g. plughw:CARD=Device,DEV=0 for a USB microphone
    sample_rate: int = Field(default=16000, ge=8000)  # what whisper wants

    # The teacher's button says who and when to start; the pause at the end
    # of a sentence says when to stop. A fixed six seconds on the Pi cut one
    # child off mid-question ("How could we...") and made every short answer
    # wait out the rest of the clip. record_seconds is now the longest turn.
    record_seconds: float = Field(default=15.0, gt=0)
    # Stop this long after the voice goes quiet. 0 records the full length.
    stop_after_silence_ms: int = Field(default=1200, ge=0)
    # Give up when nobody has started speaking by then.
    no_speech_seconds: float = Field(default=5.0, gt=0)
    # Some questions deserve more patience than others. A child answering
    # "what shall we learn about?" thinks mid-sentence, and on the Pi the
    # pause after "can we start topic on" ended the turn and the lesson was
    # written about nothing. Anything not named here uses the value above.
    patience_ms: dict[str, int] = Field(
        default_factory=lambda: {"topic": 2500, "name": 2000, "confirm": 1000}
    )
    chunk_ms: int = Field(default=100, ge=10)
    # How far above the room's own noise a chunk has to be to count as
    # speech, as a share of the way up to the loudest thing in the turn. A
    # share rather than a level, because the Pi's room measured 0.08 where a
    # fixed 0.03 sat below the hiss and every chunk read as speech.
    speech_fraction: float = Field(default=0.35, gt=0.0, lt=1.0)
    # A pause can be found when the loud parts stand this far above the quiet
    # ones, by either measure. Two, because a hot microphone and a quiet one
    # disagree about what a big difference is: the Pi measured 0.08 against
    # 0.20 one evening and 0.033 against 0.044 the next. Below both, the turn
    # records to the end rather than risk cutting a child off.
    min_gap_rms: float = Field(default=0.01, ge=0.0, le=1.0)
    min_gap_ratio: float = Field(default=1.25, ge=1.0)
    min_rms: float = Field(default=0.005, ge=0.0, le=1.0)
    # Samples averaged before loudness is measured: a cheap low-pass that
    # drops a microphone's hiss and keeps a voice. See Endpoint.smooth_samples.
    smooth_samples: int = Field(default=8, ge=1)
    # Subtracted, to drop a fan or a hum from under the voice. 0 is off; see
    # `audio_check --turn`, which measures which bands your room is noisy in.
    rumble_samples: int = Field(default=0, ge=0)

    # Recording while the robot talks hears the robot: "Sunlight What is the
    # green colour inside a leaf called?" was a child's answer with the next
    # question in it. Listening waits for the voice to finish, up to this.
    # The robot opening the microphone itself, rather than waiting for the
    # teacher's button. During these steps a child can just talk; everywhere
    # else a press is still what starts a turn.
    hands_free: bool = True
    hands_free_steps: list[str] = Field(default_factory=lambda: ["interaction", "teach"])
    hands_free_gap_seconds: float = Field(default=0.4, ge=0)

    wait_for_robot_seconds: float = Field(default=30.0, ge=0)
    quiet_poll_seconds: float = Field(default=0.05, gt=0)

    # Below this a clip is silence. Whisper invents words when given silence -
    # a full stop, or a stray "So, let's go." - so quiet audio is never sent.
    silence_peak: float = Field(default=0.02, ge=0.0, le=1.0)


class SpeechConfig(BaseModel):
    model_config = Strict

    wake: WakeConfig = Field(default_factory=WakeConfig)
    stt: SttConfig = Field(default_factory=SttConfig)
    tts: TtsConfig = Field(default_factory=TtsConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    speaker: SpeakerConfig = Field(default_factory=SpeakerConfig)


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
    # teach is lesson, interaction and quiz interleaved: say one idea, then
    # turn it over to the class. The three separate steps are still
    # registered, so [attendance, greeting, topic, lesson, interaction, quiz,
    # wrapup] gives back the straight read-through a syllabus wants.
    return ["attendance", "greeting", "topic", "teach", "quiz", "wrapup"]


def _default_timeouts() -> dict[str, float]:
    return {
        "attendance": 120.0,
        "greeting": 30.0,
        "lesson": 900.0,
        "teach": 1500.0,
        "topic": 120.0,
        "interaction": 420.0,
        "quiz": 300.0,
        "wrapup": 90.0,
    }


class TeachConfig(BaseModel):
    """Teaching as a conversation rather than a reading.

    One idea, then the class gets it back: a doubt invited, or a question
    asked about what was just said. Checking understanding while the idea is
    still in the room is the whole difference between a lesson and a
    recital.
    """

    model_config = Strict

    # How many segments between checks. 1 is after every idea, 2 is after
    # every other one, and a big number is a lecture with a check at the end.
    check_every: int = Field(default=1, ge=1)

    # alternate keeps a class awake: a question they must answer, then an
    # invitation to ask, then a question again.
    check_style: Literal["alternate", "doubts", "question"] = "alternate"

    # A check aimed at a named child, taken round the roster so the quiet
    # ones are asked too. false asks the room.
    name_a_child: bool = True

    # How long the class has to say something after an idea, before the
    # robot carries on. Short, because silence here is normal.
    gap_seconds: float = Field(default=2.5, ge=0)
    # ...and after it has invited doubts, which is longer because it has
    # just asked a question and should look like it means it.
    doubt_wait_seconds: float = Field(default=8.0, ge=0)
    # A check question is a quiz question, and waits like one.
    answer_wait_seconds: float = Field(default=20.0, gt=0)

    # A question from a child holds the lesson: the next idea must not be
    # spoken over the answer. This is how long that hold may last.
    answer_hold_seconds: float = Field(default=60.0, gt=0)

    # Hands free, the microphone is open almost all the time - it reopens
    # half a second after every turn. So an open microphone only holds the
    # lesson just after a check, and only for this long past the wait: long
    # enough that a slow child is not cut off, short enough that a room
    # which says nothing is not taught at a quarter speed.
    microphone_grace_seconds: float = Field(default=5.0, ge=0)

    # Said before carrying on after a child's question, so the class hears
    # the lesson resume rather than a sentence arriving out of nowhere.
    say_resuming: bool = True


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

    teach: TeachConfig = Field(default_factory=TeachConfig)

    questions_per_lesson: int = Field(default=6, ge=0)
    # How long to wait on a quiz question before moving on. A class where
    # nobody answers still has to reach the end of the lesson.
    answer_wait_seconds: float = Field(default=20.0, gt=0)
    # How long the robot waits to be told what to teach before falling back
    # to the lesson it was started with. A class that says nothing gets one.
    topic_wait_seconds: float = Field(default=75.0, gt=0)

    # Read the topic back before a whole class is written about it. On the Pi
    # "can we start topic on..." was heard, nothing in it was a subject, and
    # the writer invented the water cycle rather than admit it.
    confirm_topic: bool = True
    # How many times to ask again when nothing usable comes back.
    topic_tries: int = Field(default=2, ge=1)
    yes_phrases: list[str] = Field(
        default_factory=lambda: ["yes", "yeah", "yep", "correct", "right", "ok", "okay",
                                 "haan", "ha", "ji haan"]
    )
    no_phrases: list[str] = Field(
        default_factory=lambda: ["no", "nope", "wrong", "not that", "nahi", "nahin"]
    )

    # Said out loud to start a class, so the robot needs no screen at all.
    # Empty means it waits to be started from the teacher's page.
    start_phrases: list[str] = Field(
        default_factory=lambda: ["start the class", "hey lomas", "begin the class",
                                 "let us begin", "class shuru karo"]
    )
    start_poll_seconds: float = Field(default=1.0, gt=0)
    # How long switching the robot off waits for a lesson to put its report
    # away before closing the database underneath it.
    stop_wait_seconds: float = Field(default=5.0, gt=0)
    # How long an answer may take to be marked, and its feedback queued,
    # before the next question is asked anyway. A model that never answers
    # must not stop the quiz.
    mark_wait_seconds: float = Field(default=8.0, gt=0)
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

    # Where the face is drawn. `browser` is a tab pointed at /face/, which is
    # right on a laptop; `pygame` is a window the robot draws itself, which
    # on a Pi is the difference between several hundred megabytes of Chromium
    # and about fifty. `none` is a robot with a voice and no face.
    surface: Literal["browser", "pygame", "none"] = "browser"
    fullscreen: bool = True
    # Empty tries the desktop, then X11, then straight at the panel. Name one
    # (x11, kmsdrm, fbcon) only when that guessing gets it wrong.
    driver: str = ""

    # A volume control on the robot's own screen, because a robot whose
    # volume can only be changed from a laptop is a robot that depends on a
    # laptop. Three big targets - quieter, louder, mute - sized for a finger
    # on a 7-inch panel, and the up, down and M keys do the same.
    show_volume: bool = True
    volume_button_px: int = Field(default=64, ge=16)
    volume_margin_px: int = Field(default=18, ge=0)
    # The bar and the number fade out this long after the last press, so the
    # face is a face again rather than a control panel.
    volume_shown_seconds: float = Field(default=4.0, ge=0)


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
    # Long enough for an open camera stream to notice it is over and end
    # itself. Without it, shutdown cancels the response mid-frame and prints
    # a page of traceback at a teacher.
    drain_seconds: float = Field(default=0.3, ge=0)


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


def _default_sampling() -> dict[str, int]:
    """Keep one line in N. Vision publishes eight times a second and would
    otherwise bury the events that explain a slow answer."""
    return {"vision.tracks": 8}


class SyncConfig(BaseModel):
    """Getting what the robot recorded off the robot, without being asked.

    The traces are only useful where they can be read, and a Pi in a
    classroom is not that place. Asking a teacher to type three git commands
    after every class is asking for traces that never arrive - so the robot
    files its own, when a class ends and when it is switched off.
    """

    model_config = Strict

    enabled: bool = False
    # A plain string, not a fixed list: this is a registry, and a school that
    # writes its own filer registers it and names it here.
    backend: str = "git"   # git | none

    # Only these. A robot that commits its whole working tree is a robot that
    # commits a half-finished edit somebody left on it.
    paths: list[str] = Field(default_factory=lambda: ["data/logs"])

    # When to file. session_closed is after every class; paused is the
    # teacher pressing pause, which is usually why they want to look.
    on: list[Literal["session_closed", "paused", "shutdown", "timer"]] = Field(
        default_factory=lambda: ["session_closed", "shutdown"]
    )
    every_minutes: float = Field(default=0.0, ge=0.0)  # timer only; 0 is off

    remote: str = "origin"
    branch: str = ""            # empty pushes whatever branch is checked out
    push: bool = True           # false commits locally, for a robot offline
    message: str = "trace: {reason} {when}"
    when_format: str = "%Y-%m-%d %H:%M"
    timeout_seconds: float = Field(default=120.0, gt=0)
    # A push that has to wait for the network must not hold a class up, so
    # it happens on its own thread; this is how long shutdown waits for it.
    wait_seconds: float = Field(default=20.0, gt=0)


class CardsConfig(BaseModel):
    """Printed markers: who is holding one, and which way up.

    The cheap half of signing. A card costs a couple of milliseconds a frame
    where a hand model costs tens, and it says *who* is holding it, which a
    hand never does.
    """

    model_config = Strict

    reader: str = "aruco"          # aruco | none
    dictionary: str = "DICT_4X4_50"
    # Smaller than this in the picture and it is a speck across the room or
    # a pattern on somebody's shirt.
    min_size_px: int = Field(default=24, ge=4)
    # Seen this many reads running before it counts. A card that flashes by
    # as somebody puts their bag down is not an answer.
    hold_reads: int = Field(default=2, ge=1)

    # Held up means one thing: "I want to ask something". That is the whole
    # vocabulary on purpose - a classroom full of signs to remember is a
    # classroom learning the robot instead of the lesson.
    ask_when_upright: bool = True

    # Answering a quiz by which edge is up. Built, measured and off: answers
    # are spoken, because a child explaining their thinking out loud is the
    # point of the lesson and a letter is not. Turn it on for a class too
    # big to hear one by one.
    answering: bool = False
    answers: list[str] = Field(default_factory=lambda: ["A", "B", "C", "D"])
    print_answers: bool = False   # the letters on the printed card's edges

    # Printing. 10 cm at 720p reads out to about 3 m - measure yours with
    # `python tools/signs_check.py`.
    print_px: int = Field(default=420, ge=60)
    print_across: int = Field(default=3, ge=1)
    print_gap_px: int = Field(default=30, ge=0)
    sheet_dir: str = "data/cards"


class HandsConfig(BaseModel):
    """A hand held up, however this machine is able to see one.

    Two ways, and the robot's own CPU chose between them. `mediapipe` reads
    hand shapes with Google's pre-trained recognizer - accurate, and its
    aarch64 wheel is built for a processor with AES instructions, which a
    Raspberry Pi 4 does not have: it aborts the whole process with an
    illegal instruction.

    `raised_hand` is what the Pi runs instead. It knows one thing - a hand
    up beside a face - which is the only thing this robot needs a hand for,
    costs a few milliseconds, and needs no model and no wheel.
    """

    model_config = Strict

    reader: str = "none"           # raised_hand | mediapipe | none
    model: str = "models/gesture_recognizer.task"
    max_hands: int = Field(default=2, ge=1)
    min_score: float = Field(default=0.6, ge=0.0, le=1.0)
    # One read in this many. 1 is every read the watcher takes.
    every: int = Field(default=1, ge=1)
    # Three reads at three a second is a second of holding it there. Two was
    # a glimpse, and a classroom is full of glimpses.
    hold_reads: int = Field(default=3, ge=1)
    # How near a face a hand has to be, as a share of the frame width, to be
    # that child's hand. Beyond it, nobody owns the sign.
    near_face: float = Field(default=0.25, gt=0.0, le=1.0)

    # --- raised_hand only -------------------------------------------------
    # Where a raised hand is, relative to the face it belongs to: this much
    # of a face height above it, and this much of a face width to either
    # side. A hand goes up beside the head, not in front of the chest.
    above_face: float = Field(default=1.8, gt=0)
    beside_face: float = Field(default=0.9, ge=0)
    # How far down the sides of the head a hand still counts. Only the
    # sides: the strip under the chin is a neck, and a neck is skin that is
    # always there.
    below_face: float = Field(default=0.4, ge=0)

    # Skin, in chroma rather than brightness: Cr and Cb barely move across
    # skin tones where brightness moves a great deal, which is what makes
    # one setting work for a whole classroom.
    skin_cr: tuple[int, int] = (133, 180)
    skin_cb: tuple[int, int] = (77, 130)
    blur_px: int = Field(default=5, ge=1)

    # How big the blob has to be, as a share of the area of the face it is
    # next to. Below the first it is a finger's worth of noise; above the
    # second it is a wall.
    min_area: float = Field(default=0.12, gt=0)
    max_area: float = Field(default=1.6, gt=0)

    # A hand *goes* up. Skin colour alone says a wooden door, a beige wall
    # and a cardboard box are hands, and they are hands in every frame for
    # the whole afternoon - which is what the robot reported. Something that
    # has never moved is furniture.
    needs_motion: bool = True
    motion_threshold: int = Field(default=12, ge=1)     # grey levels
    motion_fraction: float = Field(default=0.06, gt=0)  # of the blob
    # How long a hand stays a hand after the movement that raised it. Long
    # enough to hold it still and be seen, short enough that the furniture
    # does not inherit it.
    motion_window_seconds: float = Field(default=4.0, gt=0)

    # One sign, one meaning. The recognizer knows thumbs and the rest, and
    # they stay unmapped: the only thing a hand is for here is saying "I
    # would like to ask something", and the name comes from the face beside
    # it. Everything else a child can say with their mouth.
    # Every reader's own name for the thing it saw, because the names are
    # the reader's and the meanings are the school's. raised_hand comes from
    # the reader that needs no model; Pointing_Up is what mediapipe calls a
    # finger in the air. A robot that switches reader must not go deaf.
    actions: dict[str, str] = Field(
        default_factory=lambda: {"raised_hand": "ask", "Pointing_Up": "ask"}
    )


class AskingConfig(BaseModel):
    """A child interrupting, and the robot letting them.

    The rule that makes it sound like a teacher rather than a machine: it
    finishes the sentence it is saying, then stops. Cutting off mid-word
    leaves a hole in the lesson that everybody hears.
    """

    model_config = Strict

    enabled: bool = True
    # Finish the sentence, then yield. false waits for the whole paragraph,
    # which is what a voice that cannot stream has to do anyway.
    finish_the_sentence: bool = True

    # Hands up, waiting. Beyond this the rest are asked to wait.
    queue_limit: int = Field(default=3, ge=1)
    # Interruptions allowed between one idea and the next, so a single
    # question does not eat a lesson.
    per_step: int = Field(default=2, ge=0)
    # The same child cannot interrupt again within this. The quiet ones
    # never get a turn otherwise.
    cooldown_seconds: float = Field(default=90.0, ge=0)
    listen_seconds: float = Field(default=12.0, gt=0)


class SignsConfig(BaseModel):
    """Saying something without saying anything.

    A room of forty children and one microphone is the problem this is for:
    a card held up is an answer from a named child with nothing to
    transcribe, and a hand held up is a question asked without shouting.
    """

    model_config = Strict

    enabled: bool = False
    # Which camera. Empty follows the one recognition uses.
    source: str = ""
    # Reads a second. Cards are cheap; this is mostly the hand model's bill.
    fps: float = Field(default=4.0, gt=0)
    downscale_width: int = Field(default=640, ge=160)
    # How long a sign counts for after it was last seen.
    remember_seconds: float = Field(default=3.0, gt=0)
    join_timeout_seconds: float = Field(default=2.0, gt=0)

    cards: CardsConfig = Field(default_factory=CardsConfig)
    hands: HandsConfig = Field(default_factory=HandsConfig)
    asking: AskingConfig = Field(default_factory=AskingConfig)


class TraceConfig(BaseModel):
    """A timeline of one run, written to a file for reading elsewhere.

    Off by default. Turned on for a session when something feels slow, then
    pushed - measuring first is the only way to tell a real improvement from
    a believed one.
    """

    model_config = Strict

    enabled: bool = False
    directory: str = "data/logs"
    name_format: str = "%Y-%m-%d_%H%M"

    # Bounded, drop-oldest, counted. A tool for finding slowness must never
    # become the slowness.
    queue_size: int = Field(default=4096, ge=16)
    shutdown_seconds: float = Field(default=5.0, gt=0)

    sample_seconds: float = Field(default=1.0, ge=0.0)  # 0 turns sampling off
    processes: bool = True
    top_processes: int = Field(default=15, ge=1)

    sample_every: dict[str, int] = Field(default_factory=_default_sampling)
    # Timing is what this is for. A few payloads are large and add nothing.
    payload_exclude: list[str] = Field(default_factory=lambda: ["vision.tracks"])

    # Endless responses have no duration worth recording - the video stream
    # would read as one request lasting the whole run.
    http_skip: list[str] = Field(default_factory=lambda: ["/camera.mjpeg"])


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
    # Steps this agent may speak in. Empty means any. For an agent that acts
    # on its own initiative rather than when asked.
    during: list[str] = Field(default_factory=list)


def _default_agent_settings() -> dict[str, AgentConfig]:
    return {
        "tutor": AgentConfig(prompt="tutor"),
        "quizmaster": AgentConfig(prompt="quizmaster", prompts={"mark": "marking", "right": "answer_right",
                                                                 "wrong": "answer_wrong"}),
        "narrator": AgentConfig(prompt="narrator"),
        "engagement": AgentConfig(prompt="nudge", during=["lesson"]),
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


class AuthorConfig(BaseModel):
    """Writing a lesson for a topic nobody wrote a pack for.

    The packs stay the first choice: they are the reviewed material. This is
    what makes the robot answer "can we learn about the solar system" with a
    lesson instead of a list of what it happens to have.
    """

    model_config = Strict

    enabled: bool = True
    prompt: str = "lesson_author"
    segments: int = Field(default=6, ge=1)
    questions: int = Field(default=6, ge=0)
    # A whole lesson plus its questions. Too small and the JSON arrives cut
    # in half, which is what "broken JSON" in front of a class means.
    max_tokens: int = Field(default=4000, ge=0)  # 0 inherits llm.max_tokens
    # Ask the endpoint itself for JSON where it supports it. Ignored by those
    # that do not.
    json_mode: bool = True

    # What a child says is a sentence, not a topic: "my name is Akshay, so
    # today we want to learn about machine learning" is a lesson on machine
    # learning. These are taken off the front and the back.
    topic_lead_ins: list[str] = Field(
        default_factory=lambda: [
            "today we want to learn about", "today we want to learn",
            "we want to learn about", "i want to learn about", "can we learn about",
            "can you teach me about", "teach me about", "tell me about",
            "let us learn about", "lets learn about", "let's learn about",
            "can we start a new lesson on", "can we start a lesson on",
            "start a new lesson on", "new lesson on", "a lesson on", "lesson on",
            "can we do a class on", "class on", "can we study",
            "i would like to learn about", "we will learn about", "learn about",
            "the topic is", "topic is", "about",
        ]
    )
    topic_tail_offs: list[str] = Field(
        default_factory=lambda: [
            "so let us go ahead and see", "so lets go ahead and see",
            "so let's go ahead and see", "let us go ahead and see",
            "lets go ahead and see", "let's go ahead and see",
            "and see", "please", "thank you", "ok", "okay", "so", "then", "now",
        ]
    )
    # "my name is Akshay, today we want to learn about machine learning"
    # arrived as a lesson on "akshay machine learning". The cue and the name
    # after it come off together.
    topic_name_cues: list[str] = Field(
        default_factory=lambda: ["my name is", "i am", "i'm", "this is", "mera naam", "मेरा नाम"]
    )
    topic_max_words: int = Field(default=8, ge=1)

    # Words that are only ever the asking. "can we start topic on" survived
    # every lead-in above, went to the writer as a subject, and came back as
    # a lesson on the water cycle that nobody had asked for. A topic made of
    # nothing but these is not a topic, and the robot asks again.
    topic_fillers: list[str] = Field(
        default_factory=lambda: [
            "a", "an", "the", "on", "of", "about", "can", "we", "you", "i", "us",
            "start", "starting", "started", "do", "does", "let", "lets", "please",
            "topic", "lesson", "class", "today", "now", "next", "new", "some",
            "something", "anything", "and", "so", "then", "is", "it", "that", "this",
            "want", "wanted", "like", "would", "should", "shall", "learn", "learning",
            "teach", "teaching", "study", "know", "tell", "talk", "ok", "okay", "hi",
        ]
    )
    # A subject shorter than this is a stray syllable, not a lesson.
    topic_min_letters: int = Field(default=3, ge=1)

    # Written lessons are kept here, not in content/: those are reviewed and
    # these are not. The same topic tomorrow then costs nothing, and works
    # with the internet down.
    cache_dir: str = "data/lessons"

    # How long the class waits while one is written before giving up and
    # teaching the lesson it already had.
    timeout_seconds: float = Field(default=45.0, gt=0)


class ContentConfig(BaseModel):
    model_config = Strict

    language: str = "en"
    grade: str = "6"
    subject: str = "science"
    vocabulary_level: Literal["primary", "middle", "secondary"] = "middle"
    pack_path: str = "content"
    default_topic: str = "photosynthesis"
    author: AuthorConfig = Field(default_factory=AuthorConfig)


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
    trace: TraceConfig = Field(default_factory=TraceConfig)
    sync: SyncConfig = Field(default_factory=SyncConfig)
    signs: SignsConfig = Field(default_factory=SignsConfig)
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)

    @property
    def is_debug(self) -> bool:
        return self.runtime.mode == "debug"

    @property
    def active_org_id(self) -> str:
        """Debug runs write to a scratch tenant so bench testing never lands
        in a real school's data."""
        return self.tenancy.scratch_org_id if self.is_debug else self.tenancy.org_id
