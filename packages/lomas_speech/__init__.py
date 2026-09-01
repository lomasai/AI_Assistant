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
from lomas_speech.player import Player, wrap_pcm
from lomas_speech.recorder import Recorder
from lomas_speech.tts import TTS_ENGINES, TextToSpeech
from lomas_speech.types import SpeechHandle, Transcript, WakeEvent
from lomas_speech.wake import WAKE_WORDS, WakeWord

from lomas_speech import stts as _stts  # noqa: F401
from lomas_speech import ttss as _ttss  # noqa: F401
from lomas_speech import wakes as _wakes  # noqa: F401

WAKE_WORDS.discover("lomas_speech.wakes")
STT_ENGINES.discover("lomas_speech.stts")
TTS_ENGINES.discover("lomas_speech.ttss")

__all__ = [
    "STT_ENGINES",
    "TTS_ENGINES",
    "WAKE_WORDS",
    "AudioConfig",
    "AudioInput",
    "AudioInputConfig",
    "DuplexGate",
    "InputSet",
    "Player",
    "Recorder",
    "wrap_pcm",
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
