from lomas_core.schema import (
    AudioConfig,
    AudioInputConfig,
    SpeechConfig,
    SttConfig,
    TtsConfig,
    WakeConfig,
)
from lomas_speech.devices import AudioInput, InputSet
from lomas_speech.duplex import DuplexGate
from lomas_speech.stt import STT_ENGINES, SpeechToText
from lomas_speech.player import Player, scale_pcm, wrap_pcm
from lomas_speech.recorder import Recorder
from lomas_speech.tts import TTS_ENGINES, TextToSpeech
from lomas_speech.types import SpeechHandle, Transcript, WakeEvent
from lomas_speech.volume import VOLUME_CONTROLS, VolumeControl, volume_control
from lomas_speech.wake import WAKE_WORDS, WakeWord

from lomas_speech import stts as _stts  # noqa: F401
from lomas_speech import ttss as _ttss  # noqa: F401
from lomas_speech import volumes as _volumes  # noqa: F401
from lomas_speech import wakes as _wakes  # noqa: F401

WAKE_WORDS.discover("lomas_speech.wakes")
VOLUME_CONTROLS.discover("lomas_speech.volumes")
STT_ENGINES.discover("lomas_speech.stts")
TTS_ENGINES.discover("lomas_speech.ttss")

__all__ = [
    "STT_ENGINES",
    "VOLUME_CONTROLS",
    "TTS_ENGINES",
    "WAKE_WORDS",
    "AudioConfig",
    "AudioInput",
    "AudioInputConfig",
    "DuplexGate",
    "InputSet",
    "Player",
    "Recorder",
    "scale_pcm",
    "wrap_pcm",
    "VolumeControl",
    "volume_control",
    "SpeechConfig",
    "SpeechHandle",
    "SpeechToText",
    "SttConfig",
    "TextToSpeech",
    "Transcript",
    "TtsConfig",
    "WakeConfig",
    "WakeEvent",
    "WakeWord",
]
