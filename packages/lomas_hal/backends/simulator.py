from __future__ import annotations

import threading

from lomas_core import logging as log
from lomas_core.clock import Clock, RealClock
from lomas_core.schema import HardwareConfig

from lomas_hal.backend import BACKENDS
from lomas_hal.gestures import GestureLibrary
from lomas_hal.protocol import (
    DECIDEGREES,
    Command,
    Flag,
    Telemetry,
    decode_telemetry,
    encode,
    encode_telemetry,
)
from lomas_hal.types import GestureHandle, SensorReading

CLEAR_MM = 2000
FLOOR_MM = 60
RESTING = "rest"
IDLE = 0.0
NOT_MOVING = 0.0


@BACKENDS.register("simulator")
class Simulator:
    """A robot with no robot in it.

    It builds the exact bytes it would have sent and logs them, so the frames
    on the bench are the frames on the wire. That is the whole point: when the
    ESP32 arrives, the only change is one config key, and if the board rejects
    a frame the log already shows what was in it.
    """

    def __init__(self, cfg: HardwareConfig, clock: Clock | None = None) -> None:
        self.cfg = cfg
        self.clock = clock or RealClock()
        self.log = log.get("hal")
        self.library = GestureLibrary(cfg.config_path)

        self.connected = False
        self.halted = False
        self.sent: list[bytes] = []
        self.gestures_played: list[str] = []
        self.pose = {name: joint.rest_degrees for name, joint in self.library.joints.items()}
        self.aim = (IDLE, IDLE)
        self.drive = (NOT_MOVING, NOT_MOVING)

        self._seq = 0
        self._playing: GestureHandle | None = None
        self._timer: threading.Timer | None = None
        self._lock = threading.RLock()

    # --- the interface ----------------------------------------------------

    def connect(self) -> None:
        problems = self.library.check()
        if problems:
            # Better here than as a servo buzzing against its stop in front
            # of a class.
            for problem in problems:
                self.log.error("gesture config: %s", problem)

        self.connected = True
        self._send(Command.PING)
        self._send(Command.SET_LIMITS, _limits(self.library))
        self._send(Command.SET_TELEMETRY_HZ, bytes((self.cfg.telemetry_hz,)))
        self.log.info("simulator connected, %d gestures", len(self.library.gestures))

    def gesture(self, name: str) -> GestureHandle:
        gesture = self.library.gesture(name)
        handle = GestureHandle(name=name, gesture_id=gesture.id, duration=gesture.duration)

        if self.halted:
            # Refused, not queued. A halted robot that catches up on three
            # gestures the moment it is cleared is a robot that hurts someone.
            self.log.warning("gesture %s refused: halted", name)
            handle.cancel()
            return handle

        self._send(Command.GESTURE, bytes((gesture.id, self.cfg.gesture_speed)))
        with self._lock:
            self._cancel_running()
            self._playing = handle
            self.gestures_played.append(name)

        self._finish_after(handle, gesture.duration)
        return handle

    def look_at(self, yaw: float, pitch: float) -> None:
        if self.halted:
            return
        yaw, pitch = self._within_reach(yaw, pitch)
        self.aim = (yaw, pitch)
        self._send(Command.LOOK_AT, _angles(yaw, pitch))

    def move(self, linear: float, angular: float) -> None:
        if self.halted:
            return
        self.drive = (linear, angular)
        self._send(Command.MOVE, _thousandths(linear, angular))

    def read_sensors(self) -> SensorReading:
        """Plausible, and quiet. The simulator reports a clear floor and open
        space, because inventing a cliff would train everyone to ignore it."""
        flags = Flag.CALIBRATED | (Flag.HALTED if self.halted else 0)
        telemetry = decode_telemetry(
            encode_telemetry(
                flags=int(flags),
                ultrasonic_mm=(CLEAR_MM,) * len(self.library.limits or [1]) or (CLEAR_MM,),
                cliff_mm=(FLOOR_MM,) * 4,
                battery_mv=self.cfg.simulated_battery_mv,
                uptime_ms=int(self.clock.now() * 1000),
            )
        )
        return SensorReading.from_telemetry(telemetry, self.clock.now())

    def halt(self) -> None:
        """Never queued, never conditional, and it stops what is running."""
        self.halted = True
        self._send(Command.HALT)
        with self._lock:
            self._cancel_running()
        self.log.warning("halted")

    def clear_halt(self) -> None:
        self.halted = False
        self._send(Command.CLEAR_HALT)

    def disconnect(self) -> None:
        with self._lock:
            self._cancel_running()
        self.connected = False

    # --- what the bench looks at ------------------------------------------

    def frames(self) -> list[str]:
        """The exact bytes, in hex. This is what gets compared against the
        firmware's serial log when something does not move."""
        return [frame.hex(" ") for frame in self.sent]

    def pose_at(self, name: str, when: float) -> dict[str, float]:
        return self.library.gesture(name).at(when)

    # --- internals --------------------------------------------------------

    def _send(self, command: Command, payload: bytes = b"") -> bytes:
        self._seq += 1
        frame = encode(command, self._seq, payload)
        self.sent.append(frame)
        if self.cfg.log_frames:
            self.log.debug("-> %-16s %s", command.name.lower(), frame.hex(" "))
        return frame

    def _within_reach(self, yaw: float, pitch: float) -> tuple[float, float]:
        neck_yaw = self.library.joints.get("neck_yaw")
        neck_pitch = self.library.joints.get("neck_pitch")
        return (
            neck_yaw.clamp(yaw) if neck_yaw else yaw,
            neck_pitch.clamp(pitch) if neck_pitch else pitch,
        )

    def _cancel_running(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        if self._playing is not None and not self._playing.done:
            self._playing.cancel()
        self._playing = None

    def _finish_after(self, handle: GestureHandle, seconds: float) -> None:
        if seconds <= IDLE or not self.cfg.simulate_travel_time:
            self._settle(handle)
            return
        self._timer = threading.Timer(seconds, self._settle, args=(handle,))
        self._timer.daemon = True
        self._timer.start()

    def _settle(self, handle: GestureHandle) -> None:
        with self._lock:
            if handle.cancelled:
                return
            gesture = self.library.by_id(handle.gesture_id)
            if gesture is not None and gesture.frames:
                self.pose.update(gesture.frames[-1].pose)
            handle.finish()
            if self._playing is handle:
                self._playing = None


def _angles(yaw: float, pitch: float) -> bytes:
    import struct

    from lomas_hal.protocol import LOOK_AT_FORMAT

    return struct.pack(LOOK_AT_FORMAT, int(yaw * DECIDEGREES), int(pitch * DECIDEGREES))


def _thousandths(linear: float, angular: float) -> bytes:
    import struct

    from lomas_hal.protocol import MILLI, MOVE_FORMAT

    return struct.pack(MOVE_FORMAT, int(linear * MILLI), int(angular * MILLI))


def _limits(library: GestureLibrary) -> bytes:
    """The safety thresholds, uploaded to the board at connect.

    They live in sensors.yaml and are applied on the ESP32. Python knows the
    numbers only well enough to send them; it never compares against them,
    because that comparison has to happen in real time.
    """
    import struct

    from lomas_hal.protocol import DECIDEGREES, LIMITS_FORMAT

    limits = library.limits
    return struct.pack(
        LIMITS_FORMAT,
        int(limits.get("cliff_mm", 0)),
        int(limits.get("obstacle_mm", 0)),
        int(limits.get("current_ma", 0)),
        int(limits.get("tilt_degrees", 0) * DECIDEGREES),
        int(limits.get("stall_ms", 0)),
    )
