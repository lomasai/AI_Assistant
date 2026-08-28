from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

from lomas_core.errors import LomasError
from lomas_store import TenantScope

from app.context.assembler import AgentContext, ContextAssembler

SCHEME = "lomas://"
MCP_AGENT = "mcp"
JSON_MIME = "application/json"

SESSION = f"{SCHEME}session"
LESSON = f"{SCHEME}lesson"
HISTORY = f"{SCHEME}history"
ROSTER = f"{SCHEME}roster"
STUDENT = f"{SCHEME}student"


@dataclass(frozen=True, slots=True)
class Resource:
    uri: str
    name: str
    description: str
    mime_type: str = JSON_MIME


class ContextServer:
    """The agent context, published as MCP resources.

    MCP is transport. Everything readable here comes back through the
    assembler, so an outside client sees exactly what an in-process agent
    sees - tenant scope included, applied before anything reaches this file.
    Bolting a stdio or HTTP server on top is P10's job.
    """

    def __init__(self, assembler: ContextAssembler) -> None:
        self.assembler = assembler
        self._readers: dict[str, Callable[[AgentContext], dict]] = {
            SESSION: _session,
            LESSON: _lesson,
            HISTORY: _history,
            ROSTER: _roster,
        }

    @property
    def enabled(self) -> bool:
        return self.assembler.cfg.context.mcp_enabled

    def list_resources(self) -> list[Resource]:
        prompts = self.assembler.cfg.content
        return [
            Resource(SESSION, "session", f"the running {prompts.subject} session"),
            Resource(LESSON, "lesson", "the segments taught so far"),
            Resource(HISTORY, "history", "recent questions and answers"),
            Resource(ROSTER, "roster", "who is present"),
            Resource(f"{STUDENT}/<id>", "student", "one child's profile"),
        ]

    def read(self, uri: str, scope: TenantScope, session_id: str) -> dict:
        student_id = uri.rsplit("/", maxsplit=1)[-1] if uri.startswith(f"{STUDENT}/") else ""
        base = uri if not student_id else STUDENT

        reader = self._readers.get(base, _student if student_id else None)
        if reader is None:
            known = ", ".join(sorted(self._readers) + [f"{STUDENT}/<id>"])
            raise LomasError(f"no resource '{uri}'. Known: {known}")

        ctx = self.assembler.for_agent(MCP_AGENT, scope, session_id, student_id)
        return reader(ctx)


def _session(ctx: AgentContext) -> dict:
    return {
        "session_id": ctx.session_id,
        "language": ctx.language,
        "topic": ctx.topic,
        "title": ctx.lesson_title,
        "grade": ctx.grade,
        "subject": ctx.subject,
    }


def _lesson(ctx: AgentContext) -> dict:
    return {"topic": ctx.topic, "taught_so_far": ctx.lesson_text}


def _history(ctx: AgentContext) -> dict:
    return {"turns": [asdict(turn) for turn in ctx.history]}


def _roster(ctx: AgentContext) -> dict:
    return {"present": list(ctx.present)}


def _student(ctx: AgentContext) -> dict:
    return asdict(ctx.student) if ctx.student else {}
