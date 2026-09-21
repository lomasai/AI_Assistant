from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

from lomas_core import logging as log
from lomas_core.clock import Clock
from lomas_core.contracts import SESSION_OPENED
from lomas_core.events import EventBus
from lomas_core.schema import Config

from app.speaker.resolver import RESOLVERS
from app.speaker.room import Room
from app.speaker.types import Heard, Speaker

NOBODY = Speaker()
ROSTER = "student"


@dataclass(slots=True)
class Deps:
    """What a resolver is given. Note what is missing: no orchestrator, no
    listener, no way to reach another resolver."""

    cfg: Config
    bus: EventBus
    clock: Clock
    prompts: Any
    log: Any


class SpeakerChain:
    """Works out who just spoke, by asking each configured way in turn.

    The order is `speech.speaker.resolvers`, and the first one that is sure
    wins. A teacher's tap is first because the person in the room overrules
    the robot; the robot asking out loud is last because interrupting a child
    to ask their name is the thing this is all meant to avoid.
    """

    def __init__(self, cfg: Config, bus: EventBus, clock: Clock, prompts: Any,
                 repos: dict[str, Any], room: Room | None = None, scope_of=None) -> None:
        self.cfg = cfg
        self.clock = clock
        self.repos = repos
        self.room = room
        self.scope_of = scope_of
        self.log = log.get("speaker")
        self.deps = Deps(cfg=cfg, bus=bus, clock=clock, prompts=prompts, log=self.log)

        self.resolvers = [RESOLVERS.create(name, cfg.speech.speaker)
                          for name in cfg.speech.speaker.resolvers]
        self._last: Speaker | None = None
        self._last_at = 0.0
        self._lock = threading.RLock()

        bus.subscribe(SESSION_OPENED, lambda *_: self.forget())

    def names(self) -> list[str]:
        return [resolver.name for resolver in self.resolvers]

    def resolve(self, text: str, tapped: tuple[str, str] = ("", ""),
                session_id: str = "", language: str = "") -> Speaker:
        with self._lock:
            last, last_at = self._last, self._last_at

        heard = Heard(
            text=text,
            session_id=session_id,
            language=language or self.cfg.content.language,
            tapped=tapped,
            roster=self._roster(self.scope_of() if self.scope_of else None),
            visible=self.room.visible() if self.room else [],
            mouths=self.room.mouths() if self.room else {},
            last=last,
            since_last=self.clock.now() - last_at if last else 0.0,
        )

        for resolver in self.resolvers:
            found = resolver.resolve(heard, self.deps)
            if found is None:
                continue
            spoken = Speaker(student_id=found.student_id, name=found.name or heard.named(
                found.student_id), how=found.how or resolver.name,
                text=found.text or text)
            if spoken:
                self._remember(spoken)
                self.log.debug("%s spoke, by %s", spoken.name, spoken.how)
            return spoken

        return Speaker(text=text, how="unknown")

    def forget(self) -> None:
        """A new session starts with nobody speaking. Otherwise the first
        question of the afternoon is attributed to whoever spoke last in the
        morning."""
        with self._lock:
            self._last, self._last_at = None, 0.0

    def _remember(self, spoken: Speaker) -> None:
        with self._lock:
            self._last, self._last_at = spoken, self.clock.now()

    def _roster(self, scope) -> list[dict]:
        if scope is None or ROSTER not in self.repos:
            return []
        return [{"id": row["id"], "name": row["name"]}
                for row in self.repos[ROSTER].list_for_class(scope)]
