from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from lomas_core import logging as log
from lomas_core.clock import Clock
from lomas_core.contracts import (
    AGENT_FAILED,
    ROBOT_SAY,
    SESSION_CLOSED,
    SESSION_OPENED,
    AgentFailed,
    Utterance,
)
from lomas_core.events import EventBus
from lomas_core.registry import Registry
from lomas_core.schema import AgentConfig, Config
from lomas_llm import Completion, PromptLibrary
from lomas_store import TenantScope

from app.context.assembler import AgentContext, ContextAssembler

ALL_EVENTS = "*"
STUDENT_FIELDS = ("student_id",)
NOTHING = ""


@runtime_checkable
class Agent(Protocol):
    name: str
    subscribes: list[str]

    def handle(self, event: str, payload: Any, ctx: AgentContext) -> None: ...


AGENTS: Registry[Agent] = Registry("agent")


@dataclass(slots=True)
class AgentDeps:
    """Everything an agent is given. Note what is missing: no store handle,
    no orchestrator, no other agent. Data arrives as context."""

    bus: EventBus
    clock: Clock
    prompts: PromptLibrary
    llm: Any
    repos: dict[str, Any] = field(default_factory=dict)


class BaseAgent:
    """Default lifecycle so an agent writes only the part it cares about."""

    name = NOTHING
    subscribes: list[str] = []

    def __init__(self, cfg: Config, settings: AgentConfig, deps: AgentDeps) -> None:
        self.cfg = cfg
        self.settings = settings
        self.deps = deps
        self.log = log.get(f"agent.{self.name}")

    def handle(self, event: str, payload: Any, ctx: AgentContext) -> None: ...

    def ask(self, ctx: AgentContext, role: str = NOTHING, **values: Any) -> Completion:
        prompt = self.settings.prompts.get(role, self.settings.prompt) if role else self.settings.prompt
        messages = self.deps.prompts.messages(prompt, ctx.language, **values)
        return self.deps.llm.complete(messages, language=ctx.language, **self._options())

    def say(self, ctx: AgentContext, text: str, student_name: str = NOTHING) -> None:
        if not text.strip():
            return
        self.deps.bus.publish(
            ROBOT_SAY,
            Utterance(
                text=text.strip(),
                language=ctx.language,
                session_id=ctx.session_id,
                student_name=student_name,
                reason=self.name,
            ),
        )

    def _options(self) -> dict:
        return {"max_tokens": self.settings.max_tokens} if self.settings.max_tokens else {}


class AgentRunner:
    """One subscription for all of them.

    In user mode an agent that raises is logged, reported and skipped: a
    tutor that cannot reach its provider must not end the lesson. On the
    bench it is re-raised, because a swallowed exception in debug mode is how
    you ship the bug.
    """

    def __init__(
        self,
        agents: list[Agent],
        assembler: ContextAssembler,
        bus: EventBus,
        clock: Clock,
        cfg: Config,
    ) -> None:
        self.agents = agents
        self.assembler = assembler
        self.bus = bus
        self.clock = clock
        self.cfg = cfg
        self.log = log.get("agents")
        self.failures = 0

        self.session_id = NOTHING
        self.scope: TenantScope | None = None

        bus.subscribe(SESSION_OPENED, self._on_open)
        bus.subscribe(SESSION_CLOSED, self._on_close)
        bus.subscribe(ALL_EVENTS, self._dispatch)

    def names(self) -> list[str]:
        return [agent.name for agent in self.agents]

    def _on_open(self, _event: str, opened) -> None:
        self.session_id = opened.session_id
        self.scope = TenantScope(opened.org_id, opened.school_id, opened.class_id)

    def _on_close(self, _event: str, _closed) -> None:
        self.session_id = NOTHING
        self.scope = None

    def _dispatch(self, event: str, payload: Any) -> None:
        if self.scope is None:
            return

        for agent in self.agents:
            if not any(fnmatch.fnmatchcase(event, p) for p in agent.subscribes):
                continue
            try:
                ctx = self.assembler.for_agent(
                    agent.name, self.scope, self.session_id, _student_of(payload)
                )
                agent.handle(event, payload, ctx)
            except Exception as exc:
                self._failed(agent.name, event, exc)

    def _failed(self, agent: str, event: str, exc: BaseException) -> None:
        self.failures += 1
        self.log.error("%s failed on %s: %s", agent, event, exc, exc_info=exc)
        self.bus.publish(
            AGENT_FAILED,
            AgentFailed(agent=agent, event=event, error=str(exc), at=self.clock.now()),
        )
        if self.cfg.runtime.raise_on_handler_error:
            raise exc


def _student_of(payload: Any) -> str:
    for name in STUDENT_FIELDS:
        value = getattr(payload, name, None)
        if value:
            return str(value)
    return NOTHING
