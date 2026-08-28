from __future__ import annotations

import threading
from dataclasses import dataclass, field
from itertools import count

SILENT = 0.0
_handles = count(1)


@dataclass(frozen=True, slots=True)
class WakeEvent:
    phrase: str
    confidence: float
    zone: str
    at: float


@dataclass(frozen=True, slots=True)
class Transcript:
    """`final` separates a live partial from the settled result.

    The face UI shows partials so a child can see they are being heard; the
    orchestrator only acts on a final.
    """

    text: str
    language: str
    final: bool = True
    confidence: float = 1.0
    zone: str = ""

    def __bool__(self) -> bool:
        return bool(self.text.strip())


@dataclass(slots=True)
class SpeechHandle:
    """One utterance in flight. Stopping it must be immediate - the teacher's
    pause button is the most important control in the product."""

    text: str
    language: str
    id: int = field(default_factory=lambda: next(_handles))
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
