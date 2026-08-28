from lomas_core.schema import HardwareConfig

from lomas_hal.backend import BACKENDS, HardwareBackend
from lomas_hal.gestures import Gesture, GestureLibrary, Joint, Keyframe
from lomas_hal.protocol import (
    Command,
    Error,
    Event,
    Flag,
    Frame,
    FrameError,
    Reader,
    Reply,
    Telemetry,
    crc8,
    decode,
    decode_telemetry,
    encode,
    encode_telemetry,
)
from lomas_hal.types import GestureHandle, SensorReading

from lomas_hal import backends as _backends  # noqa: F401

BACKENDS.discover("lomas_hal.backends")

__all__ = [
    "BACKENDS",
    "Command",
    "Error",
    "Event",
    "Flag",
    "Frame",
    "FrameError",
    "Gesture",
    "GestureHandle",
    "GestureLibrary",
    "HardwareBackend",
    "HardwareConfig",
    "Joint",
    "Keyframe",
    "Reader",
    "Reply",
    "SensorReading",
    "Telemetry",
    "crc8",
    "decode",
    "decode_telemetry",
    "encode",
    "encode_telemetry",
]
