from __future__ import annotations

import threading
from dataclasses import dataclass, field

from lomas_hal.protocol import Flag, Telemetry

NOTHING = 0.0


@dataclass(frozen=True, slots=True)
class SensorReading:
    """What the body knows about the room.

    `blocked`, `cliff` and `tilted` are the board's verdicts, not thresholds
    applied here. The distances are along for diagnostics.
    """

    ultrasonic_mm: tuple[int, ...] = ()
    cliff_mm: tuple[int, ...] = ()
    pitch: float = NOTHING
    roll: float = NOTHING
    yaw: float = NOTHING
    current_ma: int = 0
    battery_mv: int = 0
    flags: int = 0
    at: float = NOTHING

    @classmethod
    def from_telemetry(cls, telemetry: Telemetry, at: float) -> SensorReading:
        return cls(
            ultrasonic_mm=telemetry.ultrasonic_mm,
            cliff_mm=telemetry.cliff_mm,
            pitch=telemetry.pitch,
            roll=telemetry.roll,
            yaw=telemetry.yaw,
            current_ma=telemetry.current_ma,
            battery_mv=telemetry.battery_mv,
            flags=telemetry.flags,
            at=at,
        )

    def has(self, flag: Flag) -> bool:
        return bool(self.flags & flag)

    @property
    def halted(self) -> bool:
        return self.has(Flag.HALTED)

    @property
    def cliff(self) -> bool:
        return self.has(Flag.CLIFF)

    @property
    def tilted(self) -> bool:
        return self.has(Flag.TILT)

    @property
    def estop(self) -> bool:
        return self.has(Flag.ESTOP)

    @property
    def nearest_mm(self) -> int:
        live = [d for d in self.ultrasonic_mm if d]
        return min(live) if live else 0


@dataclass(slots=True)
class GestureHandle:
    """A movement in flight.

    `wait` exists so a caller can synchronise if it wants to, and almost
    nothing does: the robot gesturing while it talks is the point, and a
    lesson that blocks on a servo is a lesson that stutters.
    """

    name: str
    gesture_id: int
    duration: float
    _done: threading.Event = field(default_factory=threading.Event)
    _cancelled: bool = False

    @property
    def done(self) -> bool:
        return self._done.is_set()

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def finish(self) -> None:
        self._done.set()

    def cancel(self) -> None:
        self._cancelled = True
        self._done.set()

    def wait(self, timeout: float | None = None) -> bool:
        return self._done.wait(timeout)
