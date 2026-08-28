from __future__ import annotations

from typing import Iterator, Protocol, runtime_checkable

from lomas_core.registry import Registry
from lomas_llm.types import SYSTEM, Completion, Message


@runtime_checkable
class LLMProvider(Protocol):
    name: str

    def complete(self, messages: list[Message], **options) -> Completion: ...

    def stream(self, messages: list[Message], **options) -> Iterator[str]: ...


PROVIDERS: Registry[LLMProvider] = Registry("llm provider")


def split_system(messages: list[Message]) -> tuple[str, list[Message]]:
    """Anthropic wants the system prompt as its own argument; the
    OpenAI-compatible APIs want it as the first message. Splitting here keeps
    that difference inside the adapters."""
    system = "\n\n".join(m.content for m in messages if m.role == SYSTEM)
    rest = [m for m in messages if m.role != SYSTEM]
    return system, rest
