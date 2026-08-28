from __future__ import annotations

from dataclasses import dataclass

# Event names are the stable API between features. Adding one is free;
# changing a payload is a breaking change and should be treated as one.

SESSION_OPENED = "session.opened"
SESSION_CLOSED = "session.closed"


@dataclass(frozen=True, slots=True)
class SessionOpened:
    session_id: str
    org_id: str
    school_id: str
    class_id: str
    started_at: float


@dataclass(frozen=True, slots=True)
class SessionClosed:
    session_id: str
    ended_at: float
    reason: str
