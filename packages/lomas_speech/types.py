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
    # Set when the sound failed somewhere the caller cannot catch it, such as
    # a playback thread. Read after `wait`.
    error: str = ""

    # A child has their hand up. The robot finishes the sentence it is
    # saying and stops there - cutting off mid-word leaves a hole in the
    # lesson that everybody hears - and what it had not said yet is kept, so
    # the class can be picked up exactly where it was left.
    _yielding: bool = False
    left_to_say: str = ""

    @property
    def done(self) -> bool:
        return self._done.is_set()

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    @property
    def yielding(self) -> bool:
        return self._yielding

    def ask_to_yield(self) -> None:
        """Stop at the end of this sentence. An engine that cannot do that
        finishes the whole utterance, which is the same promise more
        coarsely kept."""
        self._yielding = True

    def keep(self, unsaid: str) -> None:
        self.left_to_say = unsaid.strip()

    def finish(self) -> None:
        self._done.set()

    def fail(self, error: str) -> None:
        self.error = error
        self._done.set()

    def cancel(self) -> None:
        self._cancelled = True
        self._done.set()

    def wait(self, timeout: float | None = None) -> bool:
        return self._done.wait(timeout)
