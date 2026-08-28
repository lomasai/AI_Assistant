from __future__ import annotations

import threading
from collections import deque
from typing import Iterator

from lomas_core.clock import Clock
from lomas_vision.frame import Frame
from lomas_vision.source import CameraSource

ANY_SOURCE = "*"
IDLE_SLEEP = 0.005


class _Subscriber:
    """A bounded view of the stream. Slow consumers lose old frames, never
    the newest one, and never stall capture."""

    def __init__(self, pattern: str, maxlen: int) -> None:
        self.pattern = pattern
        self.frames: deque[Frame] = deque(maxlen=maxlen)
        self.dropped = 0
        self.closed = False
        self._cond = threading.Condition()

    def wants(self, source_id: str) -> bool:
        return self.pattern in (ANY_SOURCE, source_id)

    def offer(self, frame: Frame) -> None:
        with self._cond:
            if len(self.frames) == self.frames.maxlen:
                self.dropped += 1
            self.frames.append(frame)
            self._cond.notify()

    def close(self) -> None:
        with self._cond:
            self.closed = True
            self._cond.notify_all()

    def stream(self, timeout: float) -> Iterator[Frame]:
        while True:
            with self._cond:
                while not self.frames and not self.closed:
                    self._cond.wait(timeout)
                if not self.frames and self.closed:
                    return
                frame = self.frames.popleft()
            yield frame


class FrameBus:
    """One capture thread per source, feeding every interested subscriber.

    Nothing here blocks on a consumer. A camera that is being read slowly
    still runs at full rate; the slow reader simply misses frames.
    """

    def __init__(self, sources: list[CameraSource], buffer_size: int, clock: Clock,
                 read_timeout_ms: int) -> None:
        self._sources = sources
        self._buffer_size = buffer_size
        self._clock = clock
        self._timeout = read_timeout_ms / 1000.0
        self._subs: list[_Subscriber] = []
        self._latest: dict[str, Frame] = {}
        self._captured: dict[str, int] = {s.source_id: 0 for s in sources}
        self._empty_reads: dict[str, int] = {s.source_id: 0 for s in sources}
        self._threads: list[threading.Thread] = []
        self._lock = threading.RLock()
        self._stop = threading.Event()

    def start(self) -> None:
        if self._threads:
            return
        for source in self._sources:
            source.open()
            thread = threading.Thread(
                target=self._pump, args=(source,), name=f"capture-{source.source_id}", daemon=True
            )
            thread.start()
            self._threads.append(thread)

    def stop(self) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=self._timeout * 2)
        self._threads.clear()
        for source in self._sources:
            source.close()
        with self._lock:
            for sub in self._subs:
                sub.close()

    def latest(self, source_id: str) -> Frame | None:
        with self._lock:
            return self._latest.get(source_id)

    def subscribe(self, source_id: str = ANY_SOURCE) -> Iterator[Frame]:
        sub = _Subscriber(source_id, self._buffer_size)
        with self._lock:
            self._subs.append(sub)
        try:
            yield from sub.stream(self._timeout)
        finally:
            with self._lock:
                if sub in self._subs:
                    self._subs.remove(sub)

    def stats(self) -> dict[str, dict[str, float]]:
        with self._lock:
            dropped = sum(sub.dropped for sub in self._subs)
            return {
                source.source_id: {
                    "captured": self._captured[source.source_id],
                    "empty_reads": self._empty_reads[source.source_id],
                    "subscribers": sum(1 for s in self._subs if s.wants(source.source_id)),
                    "dropped": dropped,
                }
                for source in self._sources
            }

    def _pump(self, source: CameraSource) -> None:
        while not self._stop.is_set():
            frame = source.read()
            if frame is None:
                with self._lock:
                    self._empty_reads[source.source_id] += 1
                self._clock.sleep(IDLE_SLEEP)
                continue

            with self._lock:
                self._latest[frame.source_id] = frame
                self._captured[frame.source_id] += 1
                targets = [s for s in self._subs if s.wants(frame.source_id)]

            for sub in targets:
                sub.offer(frame)
