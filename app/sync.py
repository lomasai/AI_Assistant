from __future__ import annotations

import threading
import time
from typing import Protocol, runtime_checkable

from lomas_core import logging as log
from lomas_core.clock import Clock
from lomas_core.contracts import SESSION_CLOSED, SESSION_PAUSED
from lomas_core.events import EventBus
from lomas_core.registry import Registry
from lomas_core.schema import Config

SESSION_ENDED = "session_closed"
PAUSED = "paused"
SHUTDOWN = "shutdown"
TIMER = "timer"
MINUTE = 60.0


@runtime_checkable
class Filer(Protocol):
    """Somewhere the robot's own records go."""

    @property
    def available(self) -> bool: ...

    def send(self, paths: list[str], message: str) -> str: ...

    def describe(self) -> str: ...


SYNCS: Registry[Filer] = Registry("sync backend")


class FileSync:
    """The robot filing its own traces, so nobody has to remember to.

    A trace is only worth writing where somebody will read it, and a Pi at
    the back of a classroom is not that place. Asking a teacher - or anybody
    holding a robot rather than a laptop - to type three git commands after
    every class is asking for traces that never arrive.

    So the robot does it: when a class ends, when it is paused, and when it
    is switched off. Only the configured paths are touched, never the whole
    working tree, because a robot that commits everything it finds is a robot
    that commits a half-finished edit somebody left on it.
    """

    def __init__(self, cfg: Config, bus: EventBus, clock: Clock) -> None:
        self.cfg = cfg
        self.bus = bus
        self.clock = clock
        self.log = log.get("sync")
        self.filed = 0

        SYNCS.discover("app.syncs")
        self.backend = SYNCS.create(cfg.sync.backend, cfg.sync) if cfg.sync.enabled else None
        self._working = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

        if not self.enabled:
            return
        if SESSION_ENDED in cfg.sync.on:
            bus.subscribe(SESSION_CLOSED, lambda *_: self.file(SESSION_ENDED))
        if PAUSED in cfg.sync.on:
            bus.subscribe(SESSION_PAUSED, lambda *_: self.file(PAUSED))
        if TIMER in cfg.sync.on and cfg.sync.every_minutes:
            self._thread = threading.Thread(target=self._every_so_often, name="sync",
                                            daemon=True)
            self._thread.start()
        self.log.info("filing %s to %s", ", ".join(cfg.sync.paths), self.backend.describe())

    @property
    def enabled(self) -> bool:
        return self.backend is not None and self.backend.available

    # --- when to file ------------------------------------------------------

    def file(self, reason: str) -> None:
        """Off on its own thread. A push that has to wait for a school's
        internet must not hold the end of a class up."""
        if not self.enabled:
            return
        threading.Thread(target=self._send, args=(reason,), name="sync-once",
                         daemon=True).start()

    def flush(self, reason: str = SHUTDOWN) -> None:
        """File now, and wait - within reason. This is the shutdown path, and
        a robot being switched off has to finish before the process goes."""
        if not self.enabled or (reason == SHUTDOWN and SHUTDOWN not in self.cfg.sync.on):
            return
        done = threading.Thread(target=self._send, args=(reason,), name="sync-flush",
                                daemon=True)
        done.start()
        done.join(timeout=self.cfg.sync.wait_seconds)
        if done.is_alive():
            self.log.warning("still filing after %.0fs; leaving it",
                             self.cfg.sync.wait_seconds)

    def close(self) -> None:
        self._stop.set()

    def _every_so_often(self) -> None:
        gap = self.cfg.sync.every_minutes * MINUTE
        while not self._stop.wait(gap):
            self._send(TIMER)

    # --- filing ------------------------------------------------------------

    def _send(self, reason: str) -> None:
        # One at a time. Two pushes of the same files race each other and the
        # second one fails in a way that looks like a real problem.
        if not self._working.acquire(blocking=False):
            self.log.debug("already filing; skipping %s", reason)
            return
        try:
            message = self.cfg.sync.message.format(
                reason=reason, when=time.strftime(self.cfg.sync.when_format))
            done = self.backend.send(list(self.cfg.sync.paths), message)
            if done:
                self.filed += 1
                self.log.info("%s", done)
        except Exception as exc:
            # Nothing here is worth ending a class over. A robot that cannot
            # reach a remote is a robot whose traces wait for the next class.
            self.log.warning("could not file the trace: %s", exc)
        finally:
            self._working.release()
