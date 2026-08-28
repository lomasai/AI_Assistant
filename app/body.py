from __future__ import annotations

from typing import Any

from lomas_core import logging as log
from lomas_core.clock import Clock
from lomas_core.contracts import (
    SAFETY_CLEARED,
    SAFETY_HALT,
    VISION_TRACKS,
    SafetyHalt,
)
from lomas_core.events import EventBus
from lomas_core.schema import Config
from lomas_hal import Event

CENTRE = 0.0
BOARD = "esp32"


class Body:
    """The robot's movement, driven by the same events everything else reads.

    It maps an event to a gesture through config, so making the robot bow at
    the end of a lesson is a line in default.yaml. And it works the other way
    too: when the board cuts out, this is what turns that into the safety halt
    the flow already knows how to obey.
    """

    def __init__(self, cfg: Config, bus: EventBus, clock: Clock, backend: Any) -> None:
        self.cfg = cfg
        self.bus = bus
        self.clock = clock
        self.backend = backend
        self.log = log.get("body")
        self.last_look = CENTRE
        self.played = 0

        for event in cfg.hardware.gestures:
            bus.subscribe(event, self._on_gesture)

        bus.subscribe(SAFETY_HALT, self._on_halt)
        bus.subscribe(SAFETY_CLEARED, self._on_cleared)

        if cfg.hardware.look_at_enabled:
            bus.subscribe(VISION_TRACKS, self._on_tracks)

        # The board reporting what it already did. Wiring it here rather than
        # inside the backend keeps lomas_hal free of the event bus.
        if hasattr(backend, "on_event"):
            backend.on_event = self._on_board_event

    def start(self) -> None:
        self.backend.connect()

    def stop(self) -> None:
        self.backend.disconnect()

    def sensors(self):
        return self.backend.read_sensors()

    # --- the robot moving -------------------------------------------------

    def _on_gesture(self, event: str, _payload) -> None:
        name = self.cfg.hardware.gestures.get(event)
        if not name:
            return
        self.backend.gesture(name)
        self.played += 1

    def _on_tracks(self, _event: str, seen) -> None:
        """The head follows whoever is nearest the middle of the frame.

        Deliberately dull: a head that snaps between children every tenth of a
        second is unsettling to watch and hard on the servos.
        """
        tracks = getattr(seen, "tracks", ()) or ()
        if not tracks or not seen.width:
            return

        middle = seen.width / 2
        target = min(tracks, key=lambda t: abs(t.x + t.w / 2 - middle))
        yaw = (target.x + target.w / 2 - middle) / middle * self._reach()

        if abs(yaw - self.last_look) < self.cfg.hardware.look_at_min_degrees:
            return
        self.last_look = yaw
        self.backend.look_at(yaw, CENTRE)

    def _reach(self) -> float:
        joint = self.backend.library.joints.get("neck_yaw")
        return joint.max_degrees if joint else CENTRE

    # --- safety, in both directions ---------------------------------------

    def _on_halt(self, _event: str, _payload) -> None:
        self.backend.halt()

    def _on_cleared(self, _event: str, _payload) -> None:
        clear = getattr(self.backend, "clear_halt", None)
        if clear is not None:
            clear()

    def _on_board_event(self, event: Event, detail: int) -> None:
        """The board has already stopped. This publishes what happened so the
        flow stops too, and the report says why."""
        if event is Event.GESTURE_DONE:
            return

        self.log.error("body cut out: %s", event.name.lower())
        self.bus.publish(
            SAFETY_HALT,
            SafetyHalt(
                reason=event.name.lower(),
                at=self.clock.now(),
                detail={"source": BOARD, "value": detail},
            ),
        )
