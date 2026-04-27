"""End-to-end pipeline integration.

Flow:
Audio -> STT -> Router -> LLM -> Action -> TTS
"""

from __future__ import annotations

import base64
import binascii
import copy
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

from server.cache import AsyncTTLCache
from server.decision_engine import DecisionEngineError, DecisionOutput, decision_engine
from server.router import RouteDecision, intent_router
from server.stt import STTError, stt_service
from server.tts import TTSError, tts_service


logger = logging.getLogger("server.pipeline")


class PipelineError(Exception):
    """Raised when one stage of the pipeline fails."""

    def __init__(self, stage: str, message: str) -> None:
        self.stage = stage
        self.message = message
        super().__init__(f"{stage}: {message}")


@dataclass(slots=True)
class PipelineResult:
    """Structured end-to-end pipeline output."""

    transcription: str
    route: dict[str, Any]
    decision: dict[str, Any]
    tts_text: str | None
    tts_audio_base64: str | None
    total_latency_ms: float
    cache: dict[str, bool] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-serializable payload."""
        return {
            "transcription": self.transcription,
            "route": self.route,
            "decision": self.decision,
            "tts_text": self.tts_text,
            "tts_audio_base64": self.tts_audio_base64,
            "total_latency_ms": self.total_latency_ms,
            "cache": self.cache,
            "warnings": self.warnings,
        }


@dataclass(slots=True)
class PipelineRuntimeConfig:
    """Runtime options for pipeline latency optimization and fallbacks."""

    enable_cache: bool = True
    cache_max_items: int = 256
    stt_cache_ttl_seconds: float = 300.0
    decision_cache_ttl_seconds: float = 180.0
    tts_cache_ttl_seconds: float = 900.0
    tts_fail_hard: bool = False

    @classmethod
    def from_env(cls) -> "PipelineRuntimeConfig":
        """Build runtime config from environment variables."""
        return cls(
            enable_cache=os.getenv("PIPELINE_CACHE_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"},
            cache_max_items=int(os.getenv("PIPELINE_CACHE_MAX_ITEMS", "256")),
            stt_cache_ttl_seconds=float(os.getenv("PIPELINE_STT_CACHE_TTL_SECONDS", "300")),
            decision_cache_ttl_seconds=float(os.getenv("PIPELINE_DECISION_CACHE_TTL_SECONDS", "180")),
            tts_cache_ttl_seconds=float(os.getenv("PIPELINE_TTS_CACHE_TTL_SECONDS", "900")),
            tts_fail_hard=os.getenv("PIPELINE_TTS_FAIL_HARD", "false").strip().lower() in {"1", "true", "yes", "on"},
        )


class PipelineEngine:
    """Async orchestrator for full multimodal interaction flow."""

    def __init__(self, config: PipelineRuntimeConfig | None = None) -> None:
        self.config = config or PipelineRuntimeConfig.from_env()
        self.stt_service = stt_service
        self.router = intent_router
        self.decision_engine = decision_engine
        self.tts_service = tts_service
        self._stt_cache: AsyncTTLCache[str, str] = AsyncTTLCache(
            ttl_seconds=self.config.stt_cache_ttl_seconds,
            max_items=self.config.cache_max_items,
        )
        self._decision_cache: AsyncTTLCache[str, DecisionOutput] = AsyncTTLCache(
            ttl_seconds=self.config.decision_cache_ttl_seconds,
            max_items=self.config.cache_max_items,
        )
        self._tts_cache: AsyncTTLCache[str, str] = AsyncTTLCache(
            ttl_seconds=self.config.tts_cache_ttl_seconds,
            max_items=self.config.cache_max_items,
        )

    async def run_from_audio_base64(
        self,
        audio_base64: str,
        filename: str = "audio.wav",
        memory: Any = None,
        context: dict[str, Any] | None = None,
        execute_actions: bool = True,
        synthesize_audio: bool = True,
    ) -> PipelineResult:
        """Run pipeline from base64-encoded audio input."""
        try:
            audio_bytes = base64.b64decode(audio_base64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise PipelineError("input", f"Invalid audio_base64 payload: {exc}") from exc

        return await self.run_from_audio_bytes(
            audio_bytes=audio_bytes,
            filename=filename,
            memory=memory,
            context=context,
            execute_actions=execute_actions,
            synthesize_audio=synthesize_audio,
        )

    async def run_from_audio_bytes(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        memory: Any = None,
        context: dict[str, Any] | None = None,
        execute_actions: bool = True,
        synthesize_audio: bool = True,
    ) -> PipelineResult:
        """Run full pipeline from raw audio bytes."""
        if not audio_bytes:
            raise PipelineError("input", "Audio payload is empty.")

        ctx = context or {}
        cache_hits = {"stt": False, "decision": False, "tts": False}
        warnings: list[str] = []
        start_total = time.perf_counter()
        logger.info("Pipeline started filename=%s bytes=%s", filename, len(audio_bytes))
        audio_hash = hashlib.sha256(audio_bytes).hexdigest()

        stage_start = time.perf_counter()
        stt_key = f"{filename}:{audio_hash}"
        transcription: str | None = None
        if self.config.enable_cache:
            transcription = await self._stt_cache.get(stt_key)
            if transcription is not None:
                cache_hits["stt"] = True

        if transcription is None:
            try:
                transcription = await self.stt_service.transcribe_bytes(audio_bytes=audio_bytes, filename=filename)
            except STTError as exc:
                raise PipelineError("stt", str(exc)) from exc
            if self.config.enable_cache:
                await self._stt_cache.set(stt_key, transcription)

        logger.info(
            "Pipeline STT done latency_ms=%.2f cache_hit=%s",
            (time.perf_counter() - stage_start) * 1000.0,
            cache_hits["stt"],
        )

        stage_start = time.perf_counter()
        route_decision = self.router.route(transcription)
        logger.info(
            "Pipeline route done intent=%s model=%s score=%s latency_ms=%.2f",
            route_decision.intent,
            route_decision.model,
            route_decision.score,
            (time.perf_counter() - stage_start) * 1000.0,
        )

        stage_start = time.perf_counter()
        decision_key = self._decision_cache_key(
            transcription=transcription,
            route_decision=route_decision,
            memory=memory,
            context=ctx,
            execute_actions=execute_actions,
        )

        decision: DecisionOutput | None = None
        if self.config.enable_cache:
            cached_decision = await self._decision_cache.get(decision_key)
            if cached_decision is not None:
                decision = copy.deepcopy(cached_decision)
                cache_hits["decision"] = True

        if decision is None:
            try:
                decision = await self.decision_engine.decide(
                    user_text=transcription,
                    memory=memory,
                    context=ctx,
                    execute_actions=execute_actions,
                    route_override=route_decision,
                )
            except DecisionEngineError as exc:
                raise PipelineError("decision", str(exc)) from exc

            # Cache only non-action responses to avoid replay side-effects.
            if self.config.enable_cache and decision.decision_type == "response":
                await self._decision_cache.set(decision_key, copy.deepcopy(decision))

        logger.info(
            "Pipeline decision done type=%s latency_ms=%.2f cache_hit=%s",
            decision.decision_type,
            (time.perf_counter() - stage_start) * 1000.0,
            cache_hits["decision"],
        )

        tts_text = self._select_tts_text(decision)
        tts_audio_base64: str | None = None
        if synthesize_audio and tts_text:
            stage_start = time.perf_counter()
            tts_key = hashlib.sha256(tts_text.encode("utf-8")).hexdigest()
            if self.config.enable_cache:
                cached_tts = await self._tts_cache.get(tts_key)
                if cached_tts is not None:
                    tts_audio_base64 = cached_tts
                    cache_hits["tts"] = True

            if tts_audio_base64 is None:
                try:
                    tts_audio = await self.tts_service.synthesize_to_bytes(tts_text)
                    tts_audio_base64 = base64.b64encode(tts_audio).decode("ascii")
                    if self.config.enable_cache:
                        await self._tts_cache.set(tts_key, tts_audio_base64)
                except TTSError as exc:
                    if self.config.tts_fail_hard:
                        raise PipelineError("tts", str(exc)) from exc
                    warnings.append(f"tts_fallback_text_only: {exc}")
                    logger.warning("Pipeline TTS failed, falling back to text-only response: %s", exc)

            logger.info(
                "Pipeline tts done latency_ms=%.2f cache_hit=%s",
                (time.perf_counter() - stage_start) * 1000.0,
                cache_hits["tts"],
            )

        total_latency_ms = round((time.perf_counter() - start_total) * 1000.0, 2)
        logger.info("Pipeline completed total_latency_ms=%.2f", total_latency_ms)

        return PipelineResult(
            transcription=transcription,
            route=self._route_to_dict(route_decision),
            decision=self._decision_to_dict(decision),
            tts_text=tts_text,
            tts_audio_base64=tts_audio_base64,
            total_latency_ms=total_latency_ms,
            cache=cache_hits,
            warnings=warnings,
        )

    @staticmethod
    def _select_tts_text(decision: DecisionOutput) -> str | None:
        if decision.response_text and decision.response_text.strip():
            return decision.response_text.strip()

        if decision.action_result and isinstance(decision.action_result.get("message"), str):
            message = str(decision.action_result["message"]).strip()
            if message:
                return message

        if decision.action and isinstance(decision.action.get("name"), str):
            return f"Action {decision.action['name']} processed."
        return None

    @staticmethod
    def _route_to_dict(route: RouteDecision) -> dict[str, Any]:
        return {
            "intent": route.intent,
            "model": route.model,
            "score": route.score,
            "confidence": route.confidence,
            "reasons": route.reasons,
            "features": route.features,
        }

    @staticmethod
    def _decision_to_dict(decision: DecisionOutput) -> dict[str, Any]:
        return {
            "decision_type": decision.decision_type,
            "intent": decision.intent,
            "model": decision.model,
            "response_text": decision.response_text,
            "action": decision.action,
            "action_result": decision.action_result,
            "confidence": decision.confidence,
            "reasons": decision.reasons,
            "raw_model_output": decision.raw_model_output,
        }

    @staticmethod
    def _decision_cache_key(
        transcription: str,
        route_decision: RouteDecision,
        memory: Any,
        context: dict[str, Any],
        execute_actions: bool,
    ) -> str:
        payload = {
            "transcription": transcription,
            "route_intent": route_decision.intent,
            "route_model": route_decision.model,
            "memory": PipelineEngine._stable_serialize(memory),
            "context": PipelineEngine._stable_serialize(context),
            "execute_actions": execute_actions,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()

    @staticmethod
    def _stable_serialize(value: Any) -> Any:
        try:
            # Convert to stable JSON-compatible structure.
            return json.loads(json.dumps(value, sort_keys=True, ensure_ascii=True, default=str))
        except Exception:  # noqa: BLE001
            return str(value)


pipeline_engine = PipelineEngine()
