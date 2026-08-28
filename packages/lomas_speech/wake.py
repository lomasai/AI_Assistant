from __future__ import annotations

from typing import Protocol, runtime_checkable

from lomas_core.registry import Registry
from lomas_speech.types import WakeEvent


@runtime_checkable
class WakeWord(Protocol):
    def start(self) -> None: ...

    def poll(self) -> WakeEvent | None: ...

    def stop(self) -> None: ...


WAKE_WORDS: Registry[WakeWord] = Registry("wake word engine")
