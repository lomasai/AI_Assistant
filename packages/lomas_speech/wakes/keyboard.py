from __future__ import annotations

import time
from collections import deque

from lomas_core.schema import WakeConfig
from lomas_speech.types import WakeEvent
from lomas_speech.wake import WAKE_WORDS

CERTAIN = 1.0


@WAKE_WORDS.register("keyboard")
class KeyboardWake:
    """Triggered by hand rather than by sound.

    This is what lets the whole teaching flow run in CI, and it is also how
    the debug console fires the robot when there is no microphone plugged in.
    """

    def __init__(self, cfg: WakeConfig) -> None:
        self.cfg = cfg
        self._pending: deque[WakeEvent] = deque()
        self._running = False

    def start(self) -> None:
        self._running = True

    def trigger(self, zone: str | None = None) -> WakeEvent:
        event = WakeEvent(
            phrase=self.cfg.phrase,
            confidence=CERTAIN,
            zone=zone or self.cfg.zone,
            at=time.monotonic(),
        )
        self._pending.append(event)
        return event

    def poll(self) -> WakeEvent | None:
        if not self._running or not self._pending:
            return None
        return self._pending.popleft()

    def stop(self) -> None:
        self._running = False
        self._pending.clear()
