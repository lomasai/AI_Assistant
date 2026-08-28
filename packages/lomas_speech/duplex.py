from __future__ import annotations

import threading

from lomas_core.clock import Clock
from lomas_core.schema import AudioConfig

MILLISECONDS = 1000.0
NOT_SPEAKING = None


class DuplexGate:
    """Mutes the microphones while the robot is talking.

    Proper echo cancellation is real work. Half-duplex costs nothing and
    removes the two failures that actually bite: the robot transcribing its
    own voice, and its own speech re-triggering the wake word. The tail keeps
    the gate shut for a moment after the speaker stops, because the room does
    not go silent the instant the audio does.
    """

    def __init__(self, cfg: AudioConfig, clock: Clock) -> None:
        self.cfg = cfg
        self.clock = clock
        self._speaking = False
        self._ended_at: float | None = NOT_SPEAKING
        self._lock = threading.RLock()

    def on_speech_start(self) -> None:
        with self._lock:
            self._speaking = True
            self._ended_at = NOT_SPEAKING

    def on_speech_end(self) -> None:
        with self._lock:
            self._speaking = False
            self._ended_at = self.clock.now()

    def is_muted(self) -> bool:
        if not self.cfg.half_duplex:
            return False
        with self._lock:
            if self._speaking:
                return True
            if self._ended_at is NOT_SPEAKING:
                return False
            elapsed = self.clock.now() - self._ended_at
        return elapsed < (self.cfg.tail_ms / MILLISECONDS)
