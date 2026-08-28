"""The contract between the Pi and the ESP32-S3.

Both sides implement this file. Python reads it; the firmware mirrors it in
`firmware/esp32/protocol.h`. If the two ever disagree the robot moves in ways
nobody asked for, so a test compares them field by field.

Frame layout, little-endian throughout:

    A5 5A | VER | SEQ | CMD | LEN | PAYLOAD (LEN bytes) | CRC8

CRC8 covers VER through the end of PAYLOAD - not the start bytes, which carry
no information, and not itself. Polynomial 0x07, no reflection, zero init:
eight lines of C, no table, which is what you want in an interrupt.

The division of labour this encodes is not negotiable. The Pi sends intent -
"do the namaste gesture", "look at this yaw" - and the ESP32 does execution:
keyframe interpolation, sensor polling, and the e-stop, cliff and tilt
cut-outs. A garbage collection pause in Python must never become a servo
glitch or a missed cliff sensor, so nothing here lets Python be in that loop.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import IntEnum

VERSION = 1

START_A = 0xA5
START_B = 0x5A
START = bytes((START_A, START_B))

HEADER_FORMAT = "<BBBB"  # version, seq, command, length
HEADER_SIZE = 4
FRAME_OVERHEAD = len(START) + HEADER_SIZE + 1  # plus the trailing CRC
MAX_PAYLOAD = 255
SEQ_WRAP = 256

CRC_POLYNOMIAL = 0x07
CRC_INIT = 0x00
CRC_MASK = 0xFF
CRC_TOP_BIT = 0x80
CRC_BITS = 8

# Angles and distances travel as integers. Floats on a wire are a portability
# argument nobody wins, and a tenth of a degree is finer than any servo here.
DECIDEGREES = 10.0
MILLI = 1000.0


class Command(IntEnum):
    """Pi to ESP32. Intent only."""

    PING = 0x01
    SET_LIMITS = 0x02  # the safety thresholds, uploaded at connect
    SET_TELEMETRY_HZ = 0x03

    GESTURE = 0x10  # gesture id + speed. The firmware owns the keyframes.
    LOOK_AT = 0x11  # yaw, pitch in decidegrees
    MOVE = 0x12  # linear, angular, thousandths of full scale
    STOP_MOTION = 0x13

    HALT = 0x20  # immediate, jumps every queue, always honoured
    CLEAR_HALT = 0x21


class Reply(IntEnum):
    """ESP32 to Pi."""

    ACK = 0x80
    NACK = 0x81
    TELEMETRY = 0x90
    EVENT = 0x91  # the board acting on its own: e-stop, cliff, tilt, stall


class Error(IntEnum):
    OK = 0x00
    BAD_CRC = 0x01
    BAD_LENGTH = 0x02
    UNKNOWN_COMMAND = 0x03
    UNKNOWN_GESTURE = 0x04
    OUT_OF_RANGE = 0x05
    BUSY = 0x06
    HALTED = 0x07  # refused because the board is latched, not because it failed
    NOT_CALIBRATED = 0x08


class Flag(IntEnum):
    """Telemetry status bits. Every one of these is a decision the ESP32 has
    already taken; Python reads them and reports, it never re-decides."""

    HALTED = 1 << 0
    ESTOP = 1 << 1
    CLIFF = 1 << 2
    TILT = 1 << 3
    OVERCURRENT = 1 << 4
    STALLED = 1 << 5
    CALIBRATED = 1 << 6


class Event(IntEnum):
    """Why the board cut out, sent the moment it does rather than at the next
    telemetry tick. A cliff detected 40 ms ago is old news."""

    ESTOP_PRESSED = 0x01
    ESTOP_RELEASED = 0x02
    CLIFF_DETECTED = 0x03
    TILT_DETECTED = 0x04
    OVERCURRENT = 0x05
    STALL = 0x06
    GESTURE_DONE = 0x07

# --- fixed-width payloads --------------------------------------------------
# The counts below are the wire layout, not a preference. Changing one is a
# firmware change, which is why they are here and not in a YAML file.

ULTRASONIC_COUNT = 6  # HC-SR04 ring
CLIFF_COUNT = 4  # VL53L0X, downward

TELEMETRY_FORMAT = (
    "<B"  # flags
    "6H"  # ultrasonic, millimetres
    "4H"  # cliff, millimetres
    "3h"  # imu pitch, roll, yaw, decidegrees
    "h"  # current, milliamps
    "H"  # battery, millivolts
    "I"  # uptime, milliseconds
)
TELEMETRY_SIZE = struct.calcsize(TELEMETRY_FORMAT)

GESTURE_FORMAT = "<BB"  # gesture id, speed percent
LOOK_AT_FORMAT = "<hh"  # yaw, pitch decidegrees
MOVE_FORMAT = "<hh"  # linear, angular thousandths
ACK_FORMAT = "<BB"  # echoed seq, error code
EVENT_FORMAT = "<BH"  # event code, detail
TELEMETRY_HZ_FORMAT = "<B"
LIMITS_FORMAT = "<HHHhH"  # cliff mm, obstacle mm, current mA, tilt ddeg, stall ms

FULL_SPEED = 100
MIN_SPEED = 1


def crc8(data: bytes) -> int:
    crc = CRC_INIT
    for byte in data:
        crc ^= byte
        for _ in range(CRC_BITS):
            if crc & CRC_TOP_BIT:
                crc = ((crc << 1) ^ CRC_POLYNOMIAL) & CRC_MASK
            else:
                crc = (crc << 1) & CRC_MASK
    return crc


def encode(command: Command, seq: int, payload: bytes = b"") -> bytes:
    if len(payload) > MAX_PAYLOAD:
        raise ValueError(f"payload of {len(payload)} bytes exceeds {MAX_PAYLOAD}")
    body = struct.pack(HEADER_FORMAT, VERSION, seq % SEQ_WRAP, int(command), len(payload)) + payload
    return START + body + bytes((crc8(body),))


@dataclass(frozen=True, slots=True)
class Frame:
    version: int
    seq: int
    kind: int
    payload: bytes

    @property
    def command(self) -> Command | None:
        try:
            return Command(self.kind)
        except ValueError:
            return None

    @property
    def reply(self) -> Reply | None:
        try:
            return Reply(self.kind)
        except ValueError:
            return None


class FrameError(ValueError):
    """A frame that arrived wrong. Never fatal: on a serial line noise is
    normal, and the right answer is to resynchronise, not to stop."""


def decode(raw: bytes) -> Frame:
    if len(raw) < FRAME_OVERHEAD:
        raise FrameError(f"frame of {len(raw)} bytes is shorter than the header")
    if raw[0] != START_A or raw[1] != START_B:
        raise FrameError("frame does not start with the sync bytes")

    body = raw[len(START) : -1]
    version, seq, kind, length = struct.unpack_from(HEADER_FORMAT, body)
    payload = body[HEADER_SIZE:]

    if len(payload) != length:
        raise FrameError(f"declared {length} payload bytes, carried {len(payload)}")
    if crc8(body) != raw[-1]:
        raise FrameError("checksum mismatch")
    if version != VERSION:
        raise FrameError(f"frame speaks protocol {version}, this is {VERSION}")

    return Frame(version=version, seq=seq, kind=kind, payload=payload)


class Reader:
    """Resynchronising frame reader.

    Serial lines drop bytes and start mid-frame. This hunts for the sync
    pattern rather than assuming the stream is aligned, so one bad frame
    costs one frame instead of every frame after it.
    """

    def __init__(self) -> None:
        self.buffer = bytearray()
        self.dropped = 0

    def feed(self, chunk: bytes) -> list[Frame]:
        self.buffer.extend(chunk)
        frames: list[Frame] = []

        while True:
            start = self.buffer.find(START)
            if start < 0:
                # Keep one byte back: the sync pattern may straddle two reads.
                self.dropped += max(0, len(self.buffer) - 1)
                del self.buffer[: max(0, len(self.buffer) - 1)]
                return frames

            if start:
                self.dropped += start
                del self.buffer[:start]

            if len(self.buffer) < FRAME_OVERHEAD:
                return frames

            length = self.buffer[len(START) + 3]
            total = FRAME_OVERHEAD + length
            if len(self.buffer) < total:
                return frames

            raw = bytes(self.buffer[:total])
            del self.buffer[:total]
            try:
                frames.append(decode(raw))
            except FrameError:
                self.dropped += total


@dataclass(frozen=True, slots=True)
class Telemetry:
    """One sensor block, exactly as the board sent it.

    Every flag here is a decision the ESP32 already made. Nothing in Python
    re-derives them from the distances: that loop has to be real time, and
    Python is not.
    """

    flags: int
    ultrasonic_mm: tuple[int, ...]
    cliff_mm: tuple[int, ...]
    pitch: float
    roll: float
    yaw: float
    current_ma: int
    battery_mv: int
    uptime_ms: int

    def has(self, flag: Flag) -> bool:
        return bool(self.flags & flag)

    @property
    def halted(self) -> bool:
        return self.has(Flag.HALTED)

    @property
    def safe(self) -> bool:
        return not (self.flags & (Flag.ESTOP | Flag.CLIFF | Flag.TILT | Flag.OVERCURRENT))


def encode_telemetry(
    flags: int,
    ultrasonic_mm: tuple[int, ...],
    cliff_mm: tuple[int, ...],
    pitch: float = 0.0,
    roll: float = 0.0,
    yaw: float = 0.0,
    current_ma: int = 0,
    battery_mv: int = 0,
    uptime_ms: int = 0,
) -> bytes:
    return struct.pack(
        TELEMETRY_FORMAT,
        flags,
        *_fit(ultrasonic_mm, ULTRASONIC_COUNT),
        *_fit(cliff_mm, CLIFF_COUNT),
        int(pitch * DECIDEGREES),
        int(roll * DECIDEGREES),
        int(yaw * DECIDEGREES),
        current_ma,
        battery_mv,
        uptime_ms,
    )


def decode_telemetry(payload: bytes) -> Telemetry:
    if len(payload) != TELEMETRY_SIZE:
        raise FrameError(f"telemetry is {TELEMETRY_SIZE} bytes, got {len(payload)}")

    values = struct.unpack(TELEMETRY_FORMAT, payload)
    at = 1
    ultrasonic = values[at : at + ULTRASONIC_COUNT]
    at += ULTRASONIC_COUNT
    cliff = values[at : at + CLIFF_COUNT]
    at += CLIFF_COUNT
    pitch, roll, yaw, current, battery, uptime = values[at:]

    return Telemetry(
        flags=values[0],
        ultrasonic_mm=tuple(ultrasonic),
        cliff_mm=tuple(cliff),
        pitch=pitch / DECIDEGREES,
        roll=roll / DECIDEGREES,
        yaw=yaw / DECIDEGREES,
        current_ma=current,
        battery_mv=battery,
        uptime_ms=uptime,
    )


def _fit(values: tuple[int, ...], count: int) -> list[int]:
    padded = list(values)[:count]
    return padded + [0] * (count - len(padded))
