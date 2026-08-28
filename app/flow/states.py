from __future__ import annotations

from enum import Enum


class StepResult(Enum):
    CONTINUE = "continue"
    DONE = "done"
    ABORT = "abort"


class SessionState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    HALTED = "halted"
    CLOSED = "closed"
