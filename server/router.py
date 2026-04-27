"""Core intent router for model selection.

Routing policy (from architecture.md):
- simple  -> groq
- medium  -> groq
- complex -> deepseek
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal


IntentClass = Literal["simple", "medium", "complex"]
RouteModel = Literal["groq", "deepseek"]


@dataclass(slots=True)
class RouterConfig:
    """Configuration for rule checks and scoring thresholds."""

    long_sentence_word_threshold: int = 18
    multi_part_min_clauses: int = 2
    reasoning_keywords: set[str] = field(
        default_factory=lambda: {
            "why",
            "how",
            "explain",
            "analyze",
            "analysis",
            "reason",
            "compare",
            "tradeoff",
            "pros and cons",
            "step by step",
            "strategy",
            "plan",
            "derive",
            "evaluate",
            "optimize",
        }
    )
    simple_phrases: set[str] = field(
        default_factory=lambda: {
            "hi",
            "hello",
            "hey",
            "thanks",
            "thank you",
            "ok",
            "okay",
            "good morning",
            "good night",
        }
    )


@dataclass(slots=True)
class RouteDecision:
    """Decision payload returned by the router."""

    intent: IntentClass
    model: RouteModel
    score: int
    confidence: float
    reasons: list[str]
    features: dict[str, bool]


class IntentRouter:
    """Rule + scoring based intent router."""

    def __init__(self, config: RouterConfig | None = None) -> None:
        self.config = config or RouterConfig()

    def route(self, user_input: str) -> RouteDecision:
        """Route user input to one model using fast deterministic logic."""
        text = self._normalize(user_input)
        if not text:
            return RouteDecision(
                intent="simple",
                model="groq",
                score=0,
                confidence=1.0,
                reasons=["empty_input_default_simple"],
                features={
                    "long_sentence": False,
                    "reasoning_keywords": False,
                    "multi_part_query": False,
                },
            )

        # Rule-based fast path for very short social/ack messages.
        if text in self.config.simple_phrases:
            return RouteDecision(
                intent="simple",
                model="groq",
                score=0,
                confidence=0.99,
                reasons=["rule_based_simple_phrase"],
                features={
                    "long_sentence": False,
                    "reasoning_keywords": False,
                    "multi_part_query": False,
                },
            )

        # Scoring system from architecture.md
        long_sentence = self._is_long_sentence(text)
        has_reasoning = self._has_reasoning_keywords(text)
        multi_part = self._is_multi_part_query(text)

        score = int(long_sentence) + int(has_reasoning) + int(multi_part)
        intent, model = self._map_score_to_intent_and_model(score)

        reasons: list[str] = []
        if long_sentence:
            reasons.append("long_sentence")
        if has_reasoning:
            reasons.append("reasoning_keywords")
        if multi_part:
            reasons.append("multi_part_query")
        if not reasons:
            reasons.append("default_low_complexity")

        return RouteDecision(
            intent=intent,
            model=model,
            score=score,
            confidence=self._confidence_from_score(score),
            reasons=reasons,
            features={
                "long_sentence": long_sentence,
                "reasoning_keywords": has_reasoning,
                "multi_part_query": multi_part,
            },
        )

    async def route_async(self, user_input: str) -> RouteDecision:
        """Async wrapper for frameworks that expect awaitable router calls."""
        return self.route(user_input=user_input)

    def _is_long_sentence(self, text: str) -> bool:
        return len(text.split()) >= self.config.long_sentence_word_threshold

    def _has_reasoning_keywords(self, text: str) -> bool:
        return any(keyword in text for keyword in self.config.reasoning_keywords)

    def _is_multi_part_query(self, text: str) -> bool:
        question_marks = text.count("?")
        clause_splits = re.split(r"\b(and|also|then|after that|plus)\b|[,;]", text)
        clause_count = len([part for part in clause_splits if part and part.strip() and part.strip() not in {"and", "also", "then", "after that", "plus"}])
        return question_marks > 1 or clause_count >= self.config.multi_part_min_clauses

    @staticmethod
    def _map_score_to_intent_and_model(score: int) -> tuple[IntentClass, RouteModel]:
        if score >= 3:
            return "complex", "deepseek"
        if score == 2:
            return "medium", "groq"
        return "simple", "groq"

    @staticmethod
    def _confidence_from_score(score: int) -> float:
        if score >= 3:
            return 0.92
        if score == 2:
            return 0.82
        if score == 1:
            return 0.8
        return 0.95

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(text.lower().strip().split())


intent_router = IntentRouter()
