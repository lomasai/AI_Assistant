from __future__ import annotations

from typing import Protocol, runtime_checkable

from lomas_core.registry import Registry


@runtime_checkable
class FaceSurface(Protocol):
    """Somewhere the robot's face appears.

    A browser tab is one. A window drawn on the Pi's own panel is another,
    and a ring of LEDs on a cheaper robot would be a third - which is why
    this is a registry and not an `if`.
    """

    name: str

    def start(self) -> None: ...

    def stop(self) -> None: ...


FACE_SURFACES: Registry[FaceSurface] = Registry("face surface")
