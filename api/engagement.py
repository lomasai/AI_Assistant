"""Safe engagement APIs for observable cues and supportive interventions."""

from __future__ import annotations

import asyncio
import hmac
import json
import os
from typing import Annotated, Any

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict

from server.engagement import EngagementError, InterventionChoice, ObservableSignal
from server.runtime import ApplicationRuntime


class EnableRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool


class ChoiceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    choice: InterventionChoice


def build_engagement_router() -> APIRouter:
    router = APIRouter(prefix="/engagement", tags=["engagement"])

    def runtime_from(request: Request) -> ApplicationRuntime:
        runtime: ApplicationRuntime | None = getattr(request.app.state, "runtime", None)
        if runtime is None:
            raise HTTPException(status_code=503, detail="Runtime is not ready.")
        return runtime

    def require_admin(x_admin_token: Annotated[str | None, Header()] = None) -> None:
        expected = os.getenv("ADMIN_API_TOKEN", "").strip()
        if expected and not hmac.compare_digest(x_admin_token or "", expected):
            raise HTTPException(status_code=403, detail="Admin authorization required.")

    @router.get("/health")
    async def health(request: Request) -> dict[str, Any]:
        return runtime_from(request).engagement.health()

    @router.get("/sessions/{session_id}/state")
    async def current_state(session_id: str, request: Request) -> dict[str, Any]:
        return _student_safe(await runtime_from(request).engagement.current_state(session_id))

    @router.post("/sessions/{session_id}/signals")
    async def ingest_signal(
        session_id: str,
        payload: ObservableSignal,
        request: Request,
        x_admin_token: Annotated[str | None, Header()] = None,
    ) -> dict[str, Any]:
        require_admin(x_admin_token)
        try:
            return _student_safe(await runtime_from(request).engagement.ingest_signal(session_id, payload))
        except EngagementError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @router.post("/sessions/{session_id}/enable")
    async def set_enabled(
        session_id: str,
        payload: EnableRequest,
        request: Request,
        x_admin_token: Annotated[str | None, Header()] = None,
    ) -> dict[str, Any]:
        require_admin(x_admin_token)
        return _student_safe(await runtime_from(request).engagement.set_session_enabled(session_id, payload.enabled))

    @router.post("/sessions/{session_id}/choice")
    async def choose_action(session_id: str, payload: ChoiceRequest, request: Request) -> dict[str, Any]:
        return _student_safe(await runtime_from(request).engagement.handle_choice(session_id, payload.choice))

    @router.get("/sessions/{session_id}/history")
    async def history(
        session_id: str,
        request: Request,
        x_admin_token: Annotated[str | None, Header()] = None,
    ) -> dict[str, Any]:
        require_admin(x_admin_token)
        return await runtime_from(request).engagement.history(session_id)

    @router.get("/events")
    async def events(request: Request) -> StreamingResponse:
        async def stream():
            last_count = 0
            while True:
                events_payload = runtime_from(request).engagement.events()
                for item in events_payload[last_count:]:
                    yield f"event: engagement\ndata: {json.dumps(item, ensure_ascii=True)}\n\n"
                last_count = len(events_payload)
                await asyncio.sleep(0.5)

        return StreamingResponse(stream(), media_type="text/event-stream")

    return router


def _student_safe(payload: dict[str, Any]) -> dict[str, Any]:
    allowed = {"session_id", "enabled", "state", "message", "reason", "interventions_used"}
    return {key: value for key, value in payload.items() if key in allowed}
