from __future__ import annotations

import re

from lomas_core.schema import RouterConfig
from lomas_llm.types import Complexity, RouterDecision

WORD = re.compile(r"\w+")
MULTIPART = re.compile(r"[;?]|\band\b|\balso\b")
SINGLE = 1

LONG = "long"
REASONING = "reasoning"
MULTI = "multipart"


class Router:
    """Sends each question to exactly one model.

    Ported from the AI_Assistant router, which had this right: chaining models
    multiplies latency and cost for very little gain. Score, pick one, go.
    """

    def __init__(self, cfg: RouterConfig) -> None:
        self.cfg = cfg

    def score(self, text: str) -> tuple[int, list[str]]:
        reasons: list[str] = []
        total = 0

        if len(WORD.findall(text)) >= self.cfg.long_sentence_words:
            total += self.cfg.weight_length
            reasons.append(LONG)

        lowered = text.lower()
        if any(word in lowered for word in self.cfg.reasoning_keywords):
            total += self.cfg.weight_keywords
            reasons.append(REASONING)

        if len(MULTIPART.findall(lowered)) > SINGLE:
            total += self.cfg.weight_multipart
            reasons.append(MULTI)

        return total, reasons

    def classify(self, text: str) -> Complexity:
        if not self.cfg.enabled:
            return Complexity.SIMPLE
        total, _ = self.score(text)
        if total >= self.cfg.complex_at:
            return Complexity.COMPLEX
        if total >= self.cfg.medium_at:
            return Complexity.MEDIUM
        return Complexity.SIMPLE

    def choose(self, complexity: Complexity) -> str:
        return {
            Complexity.SIMPLE: self.cfg.simple_provider,
            Complexity.MEDIUM: self.cfg.medium_provider,
            Complexity.COMPLEX: self.cfg.complex_provider,
        }[complexity]

    def route(self, text: str) -> RouterDecision:
        total, reasons = self.score(text)
        complexity = self.classify(text)
        return RouterDecision(
            complexity=complexity,
            provider=self.choose(complexity),
            score=total,
            reasons=tuple(reasons),
        )
