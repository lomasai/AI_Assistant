from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from lomas_core.registry import Registry

from app.speaker.types import Heard, Speaker


@runtime_checkable
class Resolver(Protocol):
    """One way of working out who is speaking.

    Returns a Speaker when it is sure and None when it is not, so the chain
    can fall through to the next way. Order and membership are config, which
    is what makes "try mouth movement on its own" an edit rather than a
    branch.
    """

    name: str

    def resolve(self, heard: Heard, deps: Any) -> Speaker | None: ...


RESOLVERS: Registry[Resolver] = Registry("speaker resolver")
