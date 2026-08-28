from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

SYSTEM = "system"
USER = "user"
ASSISTANT = "assistant"


class Complexity(str, Enum):
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"


@dataclass(frozen=True, slots=True)
class Message:
    role: str
    content: str


@dataclass(frozen=True, slots=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True, slots=True)
class Completion:
    text: str
    provider: str
    model: str = ""
    usage: Usage = field(default_factory=Usage)
    finish_reason: str = "stop"

    def __bool__(self) -> bool:
        return bool(self.text.strip())


@dataclass(frozen=True, slots=True)
class RouterDecision:
    complexity: Complexity
    provider: str
    score: int
    reasons: tuple[str, ...] = ()
