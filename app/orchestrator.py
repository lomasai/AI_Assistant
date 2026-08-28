from __future__ import annotations

import fnmatch
from typing import Any

from lomas_core import logging as log
from lomas_core.clock import Clock
from lomas_core.contracts import (
    SAFETY_CLEARED,
    SAFETY_HALT,
    SESSION_CLOSED,
    SESSION_OPENED,
    SESSION_PAUSED,
    SESSION_RESUMED,
    SessionClosed,
    SessionOpened,
)
from lomas_core.events import EventBus, to_plain
from lomas_core.schema import Config
from lomas_llm import PromptLibrary
from lomas_store import TenantScope

from app.content import ContentLibrary
from app.flow.machine import Machine
from app.flow.states import SessionState
from app.session import SessionContext

CLOSED = "closed"
HALTED = "halted"
ALL_EVENTS = "*"


class Orchestrator:
    """Owns a session from open to close.

    It knows about steps, events and the database. It knows nothing about
    cameras, speech engines or AI vendors - those reached it through the
    container as interfaces and it cannot tell which ones it got.
    """

    def __init__(
        self,
        cfg: Config,
        bus: EventBus,
        clock: Clock,
        machine: Machine,
        repos: dict[str, Any],
        prompts: PromptLibrary,
        llm: Any,
        content: ContentLibrary,
    ) -> None:
        self.cfg = cfg
        self.bus = bus
        self.clock = clock
        self.machine = machine
        self.repos = repos
        self.prompts = prompts
        self.llm = llm
        self.content = content
        self.log = log.get("session")
        self.ctx: SessionContext | None = None

        bus.subscribe(SAFETY_HALT, self._on_halt)
        bus.subscribe(SAFETY_CLEARED, self._on_cleared)

    @property
    def scope(self) -> TenantScope:
        return TenantScope(
            org_id=self.cfg.active_org_id,
            school_id=self.cfg.tenancy.school_id,
            class_id=self.cfg.tenancy.class_id,
        )

    def open_session(self, topic: str = "", language: str = "", teacher: str = "") -> SessionContext:
        scope = self.scope
        language = language or self.cfg.content.language
        topic = topic or self.cfg.content.default_topic

        pack = self.content.load(language)
        lesson = pack.lesson_for(topic)

        session_id = self.repos["session"].open(scope, language, topic, teacher)
        roster = self.repos["student"].list_for_class(scope)

        self.ctx = SessionContext(
            session_id=session_id,
            scope=scope,
            cfg=self.cfg,
            bus=self.bus,
            clock=self.clock,
            language=language,
            topic=topic,
            content=pack,
            lesson=lesson,
            repos=self.repos,
            roster=roster,
            notes={"prompts": self.prompts, "llm": self.llm},
        )

        self._record_events(session_id)
        self.bus.publish(
            SESSION_OPENED,
            SessionOpened(
                session_id=session_id,
                org_id=scope.org_id,
                school_id=scope.school_id or "",
                class_id=scope.class_id or "",
                language=language,
                topic=topic,
                started_at=self.clock.now(),
            ),
        )
        self.log.info("session open: %s, %s students on the roster", lesson.title, len(roster))
        return self.ctx

    def run(self, topic: str = "", language: str = "") -> SessionState:
        ctx = self.ctx or self.open_session(topic, language)
        state = self.machine.run(ctx)
        self.close_session(HALTED if state is SessionState.HALTED else CLOSED)
        return state

    def pause(self) -> None:
        self.machine.pause()
        self.bus.publish(SESSION_PAUSED, {"session_id": self._session_id()})
        self.log.info("paused")

    def resume(self) -> None:
        self.machine.resume()
        self.bus.publish(SESSION_RESUMED, {"session_id": self._session_id()})
        self.log.info("resumed")

    def halt(self, reason: str) -> None:
        self.machine.halt(reason)
        self.log.warning("halted: %s", reason)

    def close_session(self, reason: str = CLOSED) -> None:
        if self.ctx is None:
            return
        session_id = self.ctx.session_id
        self.repos["session"].close(self.ctx.scope, session_id, reason)
        self.bus.publish(
            SESSION_CLOSED,
            SessionClosed(session_id=session_id, ended_at=self.clock.now(), reason=reason),
        )
        self.log.info("session closed: %s", reason)
        self.ctx = None

    def _session_id(self) -> str:
        return self.ctx.session_id if self.ctx else ""

    def _on_cleared(self, _event: str, _payload) -> None:
        self.machine.clear()
        self.log.info("halt cleared")

    def _on_halt(self, _event: str, payload) -> None:
        self.halt(getattr(payload, "reason", str(payload)))

    def _record_events(self, session_id: str) -> None:
        """Every event lands in the append-only log, which is the only source
        the reports read - so a report cannot disagree with what happened.

        Vision is the exception, and it has to be: it publishes on every
        detect cycle, and a quarter of a million rows an hour would bury the
        events a report is actually made of.
        """
        scope = self.scope
        repo = self.repos["event"]
        excluded = self.cfg.runtime.log_event_exclude

        def handler(name: str, payload) -> None:
            if any(fnmatch.fnmatchcase(name, pattern) for pattern in excluded):
                return
            repo.append(scope, session_id, name, to_plain(payload))

        self.bus.subscribe(ALL_EVENTS, handler)
