from __future__ import annotations

import struct
import threading

from lomas_core import logging as log
from lomas_core.clock import Clock, RealClock
from lomas_core.errors import LomasError
from lomas_core.schema import HardwareConfig

from lomas_hal.backend import BACKENDS
from lomas_hal.gestures import GestureLibrary
from lomas_hal.protocol import (
    ACK_FORMAT,
    DECIDEGREES,
    EVENT_FORMAT,
    LIMITS_FORMAT,
    LOOK_AT_FORMAT,
    MILLI,
    MOVE_FORMAT,
    Command,
    Error,
    Event,
    Flag,
    Reader,
    Reply,
    decode_telemetry,
    encode,
)
from lomas_hal.types import GestureHandle, SensorReading

NO_PORT = "pyserial is not installed. pip install pyserial, or use hardware.backend: simulator."
READ_CHUNK = 512
IDLE_SLEEP = 0.002
NOTHING = 0.0

# The board acting on its own. Every one of these is already done by the time
# it arrives: the ESP32 cut the motors and then told us.
CUT_OUT = (Event.ESTOP_PRESSED, Event.CLIFF_DETECTED, Event.TILT_DETECTED,
           Event.OVERCURRENT, Event.STALL)


@BACKENDS.register("esp32")
class Esp32:
    """The same protocol as the simulator, over a wire.

    Reading runs on its own thread because telemetry arrives whether or not
    anybody asked, and an event saying the board hit a cliff must not wait
    behind a gesture command.
    """

    def __init__(self, cfg: HardwareConfig, clock: Clock | None = None) -> None:
        self.cfg = cfg
        self.clock = clock or RealClock()
        self.log = log.get("hal")
        self.library = GestureLibrary(cfg.config_path)

        self.connected = False
        self.halted = False
        self.reading = SensorReading()
        self.events: list[tuple[Event, int]] = []
        self.on_event = None  # set by the app so a cut-out reaches the flow

        self._port = None
        self._reader = Reader()
        self._seq = 0
        self._playing: GestureHandle | None = None
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # --- the interface ----------------------------------------------------

    def connect(self) -> None:
        try:
            import serial
        except ImportError as exc:
            raise LomasError(NO_PORT) from exc

        for problem in self.library.check():
            self.log.error("gesture config: %s", problem)

        try:
            self._port = serial.Serial(
                self.cfg.port, self.cfg.baud, timeout=self.cfg.timeout_seconds
            )
        except Exception as exc:
            raise LomasError(
                f"cannot open {self.cfg.port}: {exc}. Check the cable, or use "
                "hardware.backend: simulator."
            ) from exc

        self.connected = True
        self._stop.clear()
        self._thread = threading.Thread(target=self._pump, name="hal", daemon=True)
        self._thread.start()

        self._send(Command.PING)
        self._send(Command.SET_LIMITS, self._limits())
        self._send(Command.SET_TELEMETRY_HZ, bytes((self.cfg.telemetry_hz,)))
        self.log.info("esp32 on %s at %d baud", self.cfg.port, self.cfg.baud)

    def gesture(self, name: str) -> GestureHandle:
        gesture = self.library.gesture(name)
        handle = GestureHandle(name=name, gesture_id=gesture.id, duration=gesture.duration)

        if self.halted:
            handle.cancel()
            return handle

        # The id and a speed, nothing else. The keyframes are on the board,
        # generated from the same YAML this library read.
        self._send(Command.GESTURE, bytes((gesture.id, self.cfg.gesture_speed)))
        with self._lock:
            self._playing = handle
        return handle

    def look_at(self, yaw: float, pitch: float) -> None:
        if self.halted:
            return
        neck_yaw = self.library.joints.get("neck_yaw")
        neck_pitch = self.library.joints.get("neck_pitch")
        yaw = neck_yaw.clamp(yaw) if neck_yaw else yaw
        pitch = neck_pitch.clamp(pitch) if neck_pitch else pitch
        self._send(Command.LOOK_AT,
                   struct.pack(LOOK_AT_FORMAT, int(yaw * DECIDEGREES), int(pitch * DECIDEGREES)))

    def move(self, linear: float, angular: float) -> None:
        if self.halted:
            return
        self._send(Command.MOVE,
                   struct.pack(MOVE_FORMAT, int(linear * MILLI), int(angular * MILLI)))

    def read_sensors(self) -> SensorReading:
        """The last telemetry block, not a fresh poll.

        Asking the board and waiting would put a serial round trip inside
        whatever called this. Telemetry already arrives at telemetry_hz.
        """
        with self._lock:
            return self.reading

    def halt(self) -> None:
        self.halted = True
        self._send(Command.HALT)
        with self._lock:
            if self._playing is not None and not self._playing.done:
                self._playing.cancel()
                self._playing = None

    def clear_halt(self) -> None:
        self.halted = False
        self._send(Command.CLEAR_HALT)

    def disconnect(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.cfg.timeout_seconds * 4)
            self._thread = None
        if self._port is not None:
            try:
                self._port.close()
            finally:
                self._port = None
        self.connected = False

    # --- internals --------------------------------------------------------

    def _send(self, command: Command, payload: bytes = b"") -> bytes:
        self._seq += 1
        frame = encode(command, self._seq, payload)
        if self.cfg.log_frames:
            self.log.debug("-> %-16s %s", command.name.lower(), frame.hex(" "))
        if self._port is not None:
            self._port.write(frame)
        return frame

    def _limits(self) -> bytes:
        limits = self.library.limits
        return struct.pack(
            LIMITS_FORMAT,
            int(limits.get("cliff_mm", 0)),
            int(limits.get("obstacle_mm", 0)),
            int(limits.get("current_ma", 0)),
            int(limits.get("tilt_degrees", 0) * DECIDEGREES),
            int(limits.get("stall_ms", 0)),
        )

    def _pump(self) -> None:
        while not self._stop.is_set():
            try:
                chunk = self._port.read(READ_CHUNK) if self._port else b""
            except Exception as exc:
                self.log.error("serial read failed: %s", exc)
                return

            if not chunk:
                self.clock.sleep(IDLE_SLEEP)
                continue

            for frame in self._reader.feed(chunk):
                self._handle(frame)

    def _handle(self, frame) -> None:
        kind = frame.reply
        if kind is Reply.TELEMETRY:
            telemetry = decode_telemetry(frame.payload)
            with self._lock:
                self.reading = SensorReading.from_telemetry(telemetry, self.clock.now())
                self.halted = telemetry.halted
            return

        if kind is Reply.EVENT:
            code, detail = struct.unpack(EVENT_FORMAT, frame.payload)
            self._on_board_event(Event(code), detail)
            return

        if kind is Reply.NACK:
            seq, error = struct.unpack(ACK_FORMAT, frame.payload)
            self.log.error("frame %d refused: %s", seq, Error(error).name.lower())

    def _on_board_event(self, event: Event, detail: int) -> None:
        """The board has already acted. This reports it; it never re-decides.

        By the time a cliff event arrives the motors are stopped. Anything
        here that tried to make that judgement in Python would be deciding it
        forty milliseconds late.
        """
        self.events.append((event, detail))

        if event is Event.GESTURE_DONE:
            with self._lock:
                if self._playing is not None:
                    self._playing.finish()
                    self._playing = None
            return

        if event in CUT_OUT:
            self.halted = True
            self.log.error("board cut out: %s (%d)", event.name.lower(), detail)

        if self.on_event is not None:
            self.on_event(event, detail)
