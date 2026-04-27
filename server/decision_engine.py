"""Decision engine that combines router + LLM clients.

Input:
- user text
- memory
- context

Output:
- response OR action
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from server.actions import ActionExecutionError, action_engine
from server.router import IntentRouter, RouteDecision, intent_router


DecisionType = Literal["response", "action"]
logger = logging.getLogger("server.decision_engine")


class DecisionEngineError(Exception):
    """Raised when decision generation cannot be completed."""


class LLMClientProtocol(Protocol):
    """Protocol for pluggable async LLM clients."""

    async def generate(self, *args: Any, **kwargs: Any) -> str:
        """Return a text completion."""


class ActionExecutorProtocol(Protocol):
    """Protocol for pluggable action execution."""

    async def execute(self, action: dict[str, Any], context: dict[str, Any] | None = None):  # type: ignore[no-untyped-def]
        """Execute one action payload."""


@dataclass(slots=True)
class DecisionInput:
    """Payload expected by the decision engine."""

    user_text: str
    memory: Any = None
    context: dict[str, Any] | None = None
    execute_actions: bool = True
    route_override: RouteDecision | None = None


@dataclass(slots=True)
class DecisionOutput:
    """Structured output from the decision engine."""

    decision_type: DecisionType
    intent: str
    model: str
    response_text: str | None
    action: dict[str, Any] | None
    confidence: float
    reasons: list[str]
    raw_model_output: str | None = None
    action_result: dict[str, Any] | None = None


class DecisionEngine:
    """Central decision layer that orchestrates routing and model calls."""

    def __init__(
        self,
        router: IntentRouter | None = None,
        groq_client: LLMClientProtocol | None = None,
        deepseek_client: LLMClientProtocol | None = None,
        action_executor: ActionExecutorProtocol | None = None,
    ) -> None:
        self.router = router or intent_router
        self.groq_client = groq_client
        self.deepseek_client = deepseek_client
        self.action_executor = action_executor

    @classmethod
    def from_env(cls, router: IntentRouter | None = None) -> "DecisionEngine":
        """Build engine with optional provider clients from environment."""
        groq_client: LLMClientProtocol | None = None
        deepseek_client: LLMClientProtocol | None = None

        # Optional loading keeps this module importable in minimal environments.
        try:
            from server.llm.groq_client import create_groq_client

            groq_client = create_groq_client()
        except Exception:  # noqa: BLE001
            groq_client = None

        try:
            from server.llm.deepseek_client import create_deepseek_client

            deepseek_client = create_deepseek_client()
        except Exception:  # noqa: BLE001
            deepseek_client = None

        return cls(
            router=router,
            groq_client=groq_client,
            deepseek_client=deepseek_client,
            action_executor=action_engine,
        )

    async def decide(
        self,
        user_text: str,
        memory: Any = None,
        context: dict[str, Any] | None = None,
        execute_actions: bool = True,
        route_override: RouteDecision | None = None,
    ) -> DecisionOutput:
        """Return a response or action decision for user input."""
        return await self.decide_from_input(
            DecisionInput(
                user_text=user_text,
                memory=memory,
                context=context,
                execute_actions=execute_actions,
                route_override=route_override,
            )
        )

    async def decide_from_input(self, payload: DecisionInput) -> DecisionOutput:
        """Decision flow using route -> selected model -> parsed output."""
        normalized_text = self._normalize(payload.user_text)
        if not normalized_text:
            raise DecisionEngineError("user_text cannot be empty.")

        route = payload.route_override or self.router.route(normalized_text)
        simple_response = self._simple_phrase_response(normalized_text)
        if simple_response is not None:
            return DecisionOutput(
                decision_type="response",
                intent=route.intent,
                model="local_rule",
                response_text=simple_response,
                action=None,
                confidence=route.confidence,
                reasons=route.reasons + ["local_simple_phrase_response"],
                raw_model_output=None,
            )

        system_prompt = self._build_system_prompt()
        prompt = self._build_user_prompt(
            user_text=normalized_text,
            memory=payload.memory,
            context=payload.context or {},
            route=route,
        )

        raw_output: str | None = None
        try:
            raw_output = await self._generate_with_routed_model(route=route, prompt=prompt, system_prompt=system_prompt)
        except Exception as exc:  # noqa: BLE001
            # Fallback avoids hard failure when provider is unavailable.
            logger.warning(
                "LLM generation failed; using fallback response model=%s error_type=%s error=%s",
                route.model,
                exc.__class__.__name__,
                exc,
            )
            fallback_action = self._extract_action_fallback(normalized_text, payload.context or {})
            if fallback_action is not None:
                decision = DecisionOutput(
                    decision_type="action",
                    intent=route.intent,
                    model=f"{route.model}_fallback",
                    response_text=None,
                    action=fallback_action,
                    confidence=max(0.6, route.confidence - 0.2),
                    reasons=route.reasons + ["llm_unavailable_fallback_action"],
                    raw_model_output=None,
                )
                return await self._maybe_execute_action(
                    decision=decision,
                    context=payload.context or {},
                    execute_actions=payload.execute_actions,
                )

            return DecisionOutput(
                decision_type="response",
                intent=route.intent,
                model=f"{route.model}_fallback",
                response_text="I am temporarily unable to process that deeply. Please try again.",
                action=None,
                confidence=max(0.5, route.confidence - 0.3),
                reasons=route.reasons + ["llm_unavailable_fallback_response"],
                raw_model_output=None,
            )

        parsed = self._parse_json_object(raw_output)
        if parsed is not None:
            mode = str(parsed.get("mode", "response")).strip().lower()
            if mode == "action":
                action = parsed.get("action")
                if isinstance(action, dict) and isinstance(action.get("name"), str) and action.get("name"):
                    decision = DecisionOutput(
                        decision_type="action",
                        intent=route.intent,
                        model=route.model,
                        response_text=None,
                        action=action,
                        confidence=route.confidence,
                        reasons=route.reasons + ["model_structured_action"],
                        raw_model_output=raw_output,
                    )
                    return await self._maybe_execute_action(
                        decision=decision,
                        context=payload.context or {},
                        execute_actions=payload.execute_actions,
                    )

            response_text = parsed.get("response")
            if isinstance(response_text, str) and response_text.strip():
                return DecisionOutput(
                    decision_type="response",
                    intent=route.intent,
                    model=route.model,
                    response_text=response_text.strip(),
                    action=None,
                    confidence=route.confidence,
                    reasons=route.reasons + ["model_structured_response"],
                    raw_model_output=raw_output,
                )

        fallback_action = self._extract_action_fallback(normalized_text, payload.context or {})
        if fallback_action is not None:
            decision = DecisionOutput(
                decision_type="action",
                intent=route.intent,
                model=route.model,
                response_text=None,
                action=fallback_action,
                confidence=max(0.65, route.confidence - 0.1),
                reasons=route.reasons + ["rule_based_action_fallback"],
                raw_model_output=raw_output,
            )
            return await self._maybe_execute_action(
                decision=decision,
                context=payload.context or {},
                execute_actions=payload.execute_actions,
            )

        return DecisionOutput(
            decision_type="response",
            intent=route.intent,
            model=route.model,
            response_text=raw_output.strip() if raw_output.strip() else "I heard you.",
            action=None,
            confidence=route.confidence,
            reasons=route.reasons + ["model_raw_response"],
            raw_model_output=raw_output,
        )

    async def _maybe_execute_action(
        self,
        decision: DecisionOutput,
        context: dict[str, Any],
        execute_actions: bool,
    ) -> DecisionOutput:
        if decision.decision_type != "action" or decision.action is None:
            return decision
        if not execute_actions:
            decision.action_result = {
                "ok": False,
                "name": str(decision.action.get("name", "unknown")),
                "message": "Action execution skipped by configuration.",
                "data": {},
            }
            return decision
        if self.action_executor is None:
            decision.action_result = {
                "ok": False,
                "name": str(decision.action.get("name", "unknown")),
                "message": "No action executor configured.",
                "data": {},
            }
            decision.reasons.append("action_executor_missing")
            return decision

        try:
            action_result = await self.action_executor.execute(decision.action, context=context)
            if hasattr(action_result, "as_dict"):
                decision.action_result = action_result.as_dict()
            else:
                decision.action_result = dict(action_result)
            decision.reasons.append("action_executed")
            return decision
        except ActionExecutionError as exc:
            decision.action_result = {
                "ok": False,
                "name": str(decision.action.get("name", "unknown")),
                "message": str(exc),
                "data": {},
            }
            decision.reasons.append("action_execution_error")
            return decision
        except Exception as exc:  # noqa: BLE001
            decision.action_result = {
                "ok": False,
                "name": str(decision.action.get("name", "unknown")),
                "message": f"Unexpected action execution failure: {exc}",
                "data": {"error_type": exc.__class__.__name__},
            }
            decision.reasons.append("action_execution_error")
            return decision

    async def _generate_with_routed_model(self, route: RouteDecision, prompt: str, system_prompt: str) -> str:
        """Call the single routed model (never both)."""
        if route.model == "deepseek":
            if self.deepseek_client is None:
                raise DecisionEngineError("DeepSeek client is not configured.")
            return await self.deepseek_client.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                intent=route.intent,
            )

        if self.groq_client is None:
            raise DecisionEngineError("Groq client is not configured.")
        return await self.groq_client.generate(prompt=prompt, system_prompt=system_prompt)

    @staticmethod
    def _build_system_prompt() -> str:
        return (
            "You are a robot decision module. "
            "Return strict JSON only with keys: mode, response, action. "
            "mode must be 'response' or 'action'. "
            "If mode='action', provide action as {'name': string, 'args': object}. "
            "If mode='response', set action to null."
        )

    def _build_user_prompt(self, user_text: str, memory: Any, context: dict[str, Any], route: RouteDecision) -> str:
        memory_blob = self._serialize_for_prompt(memory)
        context_blob = self._serialize_for_prompt(context)
        return (
            f"User text:\n{user_text}\n\n"
            f"Route intent:\n{route.intent}\n\n"
            f"Memory:\n{memory_blob}\n\n"
            f"Context:\n{context_blob}\n\n"
            "Decide one output mode and respond in JSON only."
        )

    @staticmethod
    def _serialize_for_prompt(value: Any, max_chars: int = 2200) -> str:
        try:
            serialized = json.dumps(value, ensure_ascii=True, default=str, indent=2)
        except TypeError:
            serialized = str(value)
        if len(serialized) <= max_chars:
            return serialized
        return serialized[:max_chars] + "...[truncated]"

    @staticmethod
    def _parse_json_object(text: str) -> dict[str, Any] | None:
        cleaned = text.strip()
        if not cleaned:
            return None

        # Remove fenced code wrappers if present.
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
            cleaned = re.sub(r"\n?```$", "", cleaned)

        try:
            payload = json.loads(cleaned)
            return payload if isinstance(payload, dict) else None
        except json.JSONDecodeError:
            pass

        # Fallback: parse first JSON object found in free text.
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            return None
        try:
            payload = json.loads(match.group(0))
            return payload if isinstance(payload, dict) else None
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _extract_action_fallback(user_text: str, context: dict[str, Any]) -> dict[str, Any] | None:
        text = user_text.lower()

        if re.search(r"\b(remind me|set reminder|medicine)\b", text):
            time_match = re.search(r"\b(?:at|for)\s+([0-9]{1,2}(?::[0-9]{2})?\s*(?:am|pm)?)\b", text)
            reminder_time = time_match.group(1) if time_match else context.get("time")
            return {
                "name": "set_reminder",
                "args": {"message": user_text, "time": reminder_time},
            }

        if re.search(r"\b(follow me|track me|start tracking)\b", text):
            return {"name": "start_tracking", "args": {}}

        if re.search(r"\b(stop tracking|stop following)\b", text):
            return {"name": "stop_tracking", "args": {}}

        return None

    @staticmethod
    def _simple_phrase_response(user_text: str) -> str | None:
        text = user_text.lower()
        if text in {"hi", "hello", "hey", "good morning"}:
            return "Hello! How can I help you?"
        if text == "good night":
            return "Good night. Let me know if you need anything before you rest."
        if text in {"thanks", "thank you"}:
            return "You're welcome."
        if text in {"ok", "okay"}:
            return "Okay."
        return None

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(text.strip().split())


decision_engine = DecisionEngine.from_env()
