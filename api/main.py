"""FastAPI application entrypoint for the AI Robot backend."""

from __future__ import annotations

import asyncio
import base64
import binascii
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Literal

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from api.vision import build_vision_router
from server.decision_engine import DecisionEngineError, decision_engine
from server.memory.vector_db import MemoryError, memory_service
from server.pipeline import PipelineError, pipeline_engine
from server.router import intent_router
from server.stt import STTError, stt_service
from server.tts import TTSError, tts_service


LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("api")
STARTED_AT = datetime.now(timezone.utc)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Run startup and shutdown hooks for the API service."""
    logger.info("Starting AI Robot API server")
    try:
        await memory_service.initialize()
        logger.info("Memory service initialized")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Memory service initialization failed: %s", exc)

    yield

    try:
        await stt_service.close()
    except Exception:  # noqa: BLE001
        pass

    try:
        await tts_service.close()
    except Exception:  # noqa: BLE001
        pass

    try:
        if decision_engine.groq_client and hasattr(decision_engine.groq_client, "close"):
            await decision_engine.groq_client.close()
        if decision_engine.deepseek_client and hasattr(decision_engine.deepseek_client, "close"):
            await decision_engine.deepseek_client.close()
    except Exception:  # noqa: BLE001
        pass

    logger.info("Shutting down AI Robot API server")


class STTRequest(BaseModel):
    """Request model for speech-to-text endpoint."""

    model_config = ConfigDict(extra="forbid")
    audio_base64: str | None = Field(default=None, description="Base64-encoded audio bytes")
    audio_path: str | None = Field(default=None, description="Local server-side audio path")
    filename: str = Field(default="audio.wav")


class STTResponse(BaseModel):
    """Response model for speech-to-text endpoint."""

    model_config = ConfigDict(extra="forbid")
    text: str
    provider: str


class RouteRequest(BaseModel):
    """Request model for routing endpoint."""

    model_config = ConfigDict(extra="forbid")
    text: str


class RouteResponse(BaseModel):
    """Response model for routing endpoint."""

    model_config = ConfigDict(extra="forbid")
    intent: str
    model: str
    score: int
    confidence: float
    reasons: list[str]
    features: dict[str, bool]


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""

    model_config = ConfigDict(extra="forbid")
    user_text: str
    memory: Any = None
    context: dict[str, Any] = Field(default_factory=dict)
    retrieve_memory: bool = True
    memory_top_k: int = 3
    memory_recent_k: int = 8
    store_log: bool = True


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""

    model_config = ConfigDict(extra="forbid")
    decision_type: Literal["response", "action"]
    intent: str
    model: str
    response_text: str | None
    action: dict[str, Any] | None
    confidence: float
    reasons: list[str]
    raw_model_output: str | None
    action_result: dict[str, Any] | None = None


class TTSRequest(BaseModel):
    """Request model for text-to-speech endpoint."""

    model_config = ConfigDict(extra="forbid")
    text: str
    output_mode: Literal["base64", "file"] = "base64"
    output_path: str | None = None


class TTSResponse(BaseModel):
    """Response model for text-to-speech endpoint."""

    model_config = ConfigDict(extra="forbid")
    output_mode: Literal["base64", "file"]
    audio_base64: str | None = None
    output_path: str | None = None


class PipelineRequest(BaseModel):
    """Request model for full audio-to-audio pipeline endpoint."""

    model_config = ConfigDict(extra="forbid")
    audio_base64: str | None = None
    audio_path: str | None = None
    filename: str = "audio.wav"
    memory: Any = None
    context: dict[str, Any] = Field(default_factory=dict)
    execute_actions: bool = True
    synthesize_audio: bool = True


class PipelineResponse(BaseModel):
    """Response model for full pipeline endpoint."""

    model_config = ConfigDict(extra="forbid")
    transcription: str
    route: dict[str, Any]
    decision: dict[str, Any]
    tts_text: str | None
    tts_audio_base64: str | None
    total_latency_ms: float
    cache: dict[str, bool] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class MemoryStoreRequest(BaseModel):
    """Request model for memory store endpoint."""

    model_config = ConfigDict(extra="forbid")
    role: str | None = None
    message: str | None = None
    summary: str | None = None
    source: str = "api"
    metadata: dict[str, Any] = Field(default_factory=dict)
    auto_summarize_recent: bool = False
    summarize_limit: int | None = None
    use_llm_summary: bool = False


class MemoryStoreResponse(BaseModel):
    """Response model for memory store endpoint."""

    model_config = ConfigDict(extra="forbid")
    stored_log_id: int | None = None
    stored_summary_id: int | None = None
    auto_summary_result: dict[str, Any] | None = None


class MemoryRetrieveRequest(BaseModel):
    """Request model for memory retrieve endpoint."""

    model_config = ConfigDict(extra="forbid")
    query: str
    top_k: int = 3
    recent_k: int | None = None


class MemoryRetrieveResponse(BaseModel):
    """Response model for memory retrieve endpoint."""

    model_config = ConfigDict(extra="forbid")
    query: str
    recent_logs: list[dict[str, Any]]
    summary_matches: list[dict[str, Any]]


def build_health_router() -> APIRouter:
    """Build health-related endpoints."""
    router = APIRouter(prefix="/health", tags=["health"])

    @router.get("", summary="Health check")
    async def health_check() -> dict[str, str]:
        return {
            "status": "ok",
            "service": "ai-robot-api",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }

    return router


def build_system_router() -> APIRouter:
    """Build system metadata endpoints."""
    router = APIRouter(prefix="/system", tags=["system"])

    @router.get("/info", summary="Service metadata")
    async def system_info() -> dict[str, str]:
        uptime_seconds = int((datetime.now(timezone.utc) - STARTED_AT).total_seconds())
        return {
            "name": "ai-robot-api",
            "version": "0.1.0",
            "uptime_seconds": str(uptime_seconds),
        }

    return router


def build_core_router() -> APIRouter:
    """Build core AI interaction endpoints."""
    router = APIRouter(tags=["core"])

    @router.post("/stt", response_model=STTResponse, summary="Speech-to-text")
    async def stt_endpoint(payload: STTRequest) -> STTResponse:
        if payload.audio_base64 is None and payload.audio_path is None:
            raise HTTPException(status_code=422, detail="Provide either audio_base64 or audio_path.")

        try:
            if payload.audio_base64 is not None:
                audio_bytes = base64.b64decode(payload.audio_base64, validate=True)
                text = await stt_service.transcribe_bytes(audio_bytes=audio_bytes, filename=payload.filename)
            else:
                text = await stt_service.transcribe_file(payload.audio_path or "")
        except (STTError, binascii.Error) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return STTResponse(text=text, provider=stt_service.config.provider)

    @router.post("/route", response_model=RouteResponse, summary="Route input to model")
    async def route_endpoint(payload: RouteRequest) -> RouteResponse:
        decision = intent_router.route(payload.text)
        return RouteResponse(
            intent=decision.intent,
            model=decision.model,
            score=decision.score,
            confidence=decision.confidence,
            reasons=decision.reasons,
            features=decision.features,
        )

    @router.post("/chat", response_model=ChatResponse, summary="Chat decision endpoint")
    async def chat_endpoint(payload: ChatRequest) -> ChatResponse:
        resolved_memory = payload.memory
        try:
            if payload.retrieve_memory:
                retrieved = await memory_service.retrieve_relevant_context(
                    query=payload.user_text,
                    top_k=max(1, payload.memory_top_k),
                    recent_k=payload.memory_recent_k,
                )
                if resolved_memory is None:
                    resolved_memory = retrieved
                else:
                    resolved_memory = {
                        "request_memory": resolved_memory,
                        "retrieved_memory": retrieved,
                    }
        except MemoryError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        try:
            result = await decision_engine.decide(
                user_text=payload.user_text,
                memory=resolved_memory,
                context=payload.context,
            )
        except (DecisionEngineError, MemoryError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if payload.store_log:
            try:
                await memory_service.store_conversation_log("user", payload.user_text, metadata={"source": "chat_api"})
                if result.decision_type == "response" and result.response_text:
                    await memory_service.store_conversation_log(
                        "assistant",
                        result.response_text,
                        metadata={"source": "chat_api", "model": result.model},
                    )
                if result.decision_type == "action" and result.action:
                    await memory_service.store_conversation_log(
                        "assistant",
                        f"ACTION::{result.action.get('name', 'unknown')}",
                        metadata={"source": "chat_api", "action": result.action, "model": result.model},
                    )
            except MemoryError as exc:
                logger.warning("Failed to store chat logs: %s", exc)

        return ChatResponse(
            decision_type=result.decision_type,
            intent=result.intent,
            model=result.model,
            response_text=result.response_text,
            action=result.action,
            confidence=result.confidence,
            reasons=result.reasons,
            raw_model_output=result.raw_model_output,
            action_result=result.action_result,
        )

    @router.post("/tts", response_model=TTSResponse, summary="Text-to-speech")
    async def tts_endpoint(payload: TTSRequest) -> TTSResponse:
        try:
            if payload.output_mode == "file":
                if not payload.output_path:
                    raise HTTPException(status_code=422, detail="output_path is required when output_mode='file'.")
                output_path = await tts_service.synthesize_to_file(payload.text, payload.output_path)
                return TTSResponse(output_mode="file", output_path=str(output_path))

            audio_bytes = await tts_service.synthesize_to_bytes(payload.text)
            return TTSResponse(
                output_mode="base64",
                audio_base64=base64.b64encode(audio_bytes).decode("ascii"),
            )
        except TTSError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/pipeline", response_model=PipelineResponse, summary="Full audio-to-response pipeline")
    async def pipeline_endpoint(payload: PipelineRequest) -> PipelineResponse:
        if payload.audio_base64 is None and payload.audio_path is None:
            raise HTTPException(status_code=422, detail="Provide either audio_base64 or audio_path.")

        try:
            if payload.audio_base64 is not None:
                result = await pipeline_engine.run_from_audio_base64(
                    audio_base64=payload.audio_base64,
                    filename=payload.filename,
                    memory=payload.memory,
                    context=payload.context,
                    execute_actions=payload.execute_actions,
                    synthesize_audio=payload.synthesize_audio,
                )
            else:
                source = Path(payload.audio_path or "")
                if not source.exists():
                    raise HTTPException(status_code=422, detail=f"audio_path not found: {source}")
                audio_bytes = await asyncio.to_thread(source.read_bytes)
                result = await pipeline_engine.run_from_audio_bytes(
                    audio_bytes=audio_bytes,
                    filename=payload.filename,
                    memory=payload.memory,
                    context=payload.context,
                    execute_actions=payload.execute_actions,
                    synthesize_audio=payload.synthesize_audio,
                )
        except PipelineError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return PipelineResponse(**result.as_dict())

    return router


def build_memory_router() -> APIRouter:
    """Build memory management endpoints."""
    router = APIRouter(prefix="/memory", tags=["memory"])

    @router.post("/store", response_model=MemoryStoreResponse, summary="Store memory items")
    async def memory_store_endpoint(payload: MemoryStoreRequest) -> MemoryStoreResponse:
        stored_log_id: int | None = None
        stored_summary_id: int | None = None
        auto_summary_result: dict[str, Any] | None = None

        try:
            if payload.role and payload.message:
                stored_log_id = await memory_service.store_conversation_log(
                    role=payload.role,
                    message=payload.message,
                    metadata=payload.metadata,
                )
            elif payload.role or payload.message:
                raise HTTPException(status_code=422, detail="Both role and message are required for log storage.")

            if payload.summary:
                stored_summary_id = await memory_service.store_summary(
                    summary=payload.summary,
                    source=payload.source,
                    metadata=payload.metadata,
                )

            if payload.auto_summarize_recent:
                auto_summary_result = await memory_service.summarize_and_store_recent(
                    limit=payload.summarize_limit,
                    source=payload.source,
                    use_llm=payload.use_llm_summary,
                )
        except MemoryError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if stored_log_id is None and stored_summary_id is None and auto_summary_result is None:
            raise HTTPException(
                status_code=422,
                detail="Provide log fields, summary, or auto_summarize_recent=true.",
            )

        return MemoryStoreResponse(
            stored_log_id=stored_log_id,
            stored_summary_id=stored_summary_id,
            auto_summary_result=auto_summary_result,
        )

    @router.post("/retrieve", response_model=MemoryRetrieveResponse, summary="Retrieve relevant memory context")
    async def memory_retrieve_endpoint(payload: MemoryRetrieveRequest) -> MemoryRetrieveResponse:
        try:
            result = await memory_service.retrieve_relevant_context(
                query=payload.query,
                top_k=max(1, payload.top_k),
                recent_k=payload.recent_k,
            )
        except MemoryError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return MemoryRetrieveResponse(**result)

    return router


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="AI Robot API",
        version="0.1.0",
        description="Backend API for modular AI robot services",
        lifespan=lifespan,
    )

    api_v1 = APIRouter(prefix="/api/v1")
    api_v1.include_router(build_health_router())
    api_v1.include_router(build_system_router())
    app.include_router(api_v1)
    app.include_router(build_core_router())
    app.include_router(build_memory_router())
    app.include_router(build_vision_router())

    # ------------------------------------------------------------------
    # Frontend static files
    #
    # If a `frontend` directory exists at the project root (one level
    # above the `api` package), mount it at `/static` so that the web
    # application can serve JavaScript and other assets. The root path
    # ("/") is routed to deliver the `index.html` page. These routes
    # are not included in the OpenAPI schema since they are purely for
    # human-facing UI and do not represent API endpoints.
    frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
    if frontend_dir.exists():
        # Mount the directory for static assets. `html=False` ensures
        # that only explicit file requests are served (e.g., /static/script.js).
        app.mount(
            "/static",
            StaticFiles(directory=str(frontend_dir), html=False),
            name="static",
        )

        @app.get("/", include_in_schema=False)
        async def serve_frontend() -> FileResponse:
            return FileResponse(frontend_dir / "index.html")

    @app.middleware("http")
    async def request_logging_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
        """Log each request and response with latency."""
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.info(
            "Request completed method=%s path=%s status=%s duration_ms=%s",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        return response

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
        """Return a safe JSON payload for unexpected server errors."""
        if isinstance(exc, HTTPException):
            return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
        logger.exception("Unhandled exception: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )

    return app


app = create_app()
