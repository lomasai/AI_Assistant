from __future__ import annotations

from typing import Protocol, runtime_checkable

from lomas_core.registry import Registry

from lomas_hal.types import GestureHandle, SensorReading


@runtime_checkable
class HardwareBackend(Protocol):
    """Intent in, sensors out.

    Nothing on this interface takes a servo angle or a sensor threshold. The
    Pi says what it wants; the board decides how, and owns every reflex that
    has to happen inside a few milliseconds.
    """

    def connect(self) -> None: ...

    def gesture(self, name: str) -> GestureHandle: ...

    def look_at(self, yaw: float, pitch: float) -> None: ...

    def move(self, linear: float, angular: float) -> None: ...

    def read_sensors(self) -> SensorReading: ...

    def halt(self) -> None: ...

    def disconnect(self) -> None: ...


BACKENDS: Registry[HardwareBackend] = Registry("hardware backend")
