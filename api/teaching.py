"""Teaching-session API endpoints."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict

from server.teaching import (
    InvalidTransitionError,
    LessonConfig,
    StudentResponse,
    TeachingError,
    TeachingSession,
)


class CreateTeachingSessionRequest(LessonConfig):
    pass


class AnswerRequest(StudentResponse):
    pass


class TeachingSessionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session: dict[str, Any]


def build_teaching_router() -> APIRouter:
    router = APIRouter(prefix="/teaching", tags=["teaching"])

    @router.post("/sessions", response_model=TeachingSessionResponse)
    async def create_session(payload: CreateTeachingSessionRequest, request: Request) -> TeachingSessionResponse:
        session = await request.app.state.runtime.teaching.create_session(LessonConfig.model_validate(payload.model_dump()))
        return _response(session)

    @router.get("/sessions/{session_id}", response_model=TeachingSessionResponse)
    async def get_session(session_id: str, request: Request) -> TeachingSessionResponse:
        try:
            return _response(await request.app.state.runtime.teaching.get_session(session_id))
        except TeachingError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.post("/sessions/{session_id}/start", response_model=TeachingSessionResponse)
    async def start_session(session_id: str, request: Request) -> TeachingSessionResponse:
        return await _command(request, "start", session_id)

    @router.post("/sessions/{session_id}/answer", response_model=TeachingSessionResponse)
    async def answer(session_id: str, payload: AnswerRequest, request: Request) -> TeachingSessionResponse:
        try:
            session = await request.app.state.runtime.teaching.submit_answer(
                session_id,
                StudentResponse.model_validate(payload.model_dump()),
            )
            return _response(session)
        except InvalidTransitionError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except TeachingError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.post("/sessions/{session_id}/pause", response_model=TeachingSessionResponse)
    async def pause(session_id: str, request: Request) -> TeachingSessionResponse:
        return await _command(request, "pause", session_id)

    @router.post("/sessions/{session_id}/resume", response_model=TeachingSessionResponse)
    async def resume(session_id: str, request: Request) -> TeachingSessionResponse:
        return await _command(request, "resume", session_id)

    @router.post("/sessions/{session_id}/stop", response_model=TeachingSessionResponse)
    async def stop(session_id: str, request: Request) -> TeachingSessionResponse:
        return await _command(request, "stop", session_id)

    @router.get("/sessions/{session_id}/summary")
    async def summary(session_id: str, request: Request) -> dict[str, Any]:
        try:
            return (await request.app.state.runtime.teaching.summary(session_id)).model_dump(mode="json")
        except TeachingError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.get("/sessions/{session_id}/events")
    async def events(session_id: str, request: Request) -> StreamingResponse:
        async def stream():
            last_count = 0
            while True:
                events_payload = request.app.state.runtime.teaching.events(session_id)
                for item in events_payload[last_count:]:
                    yield f"event: teaching_session\ndata: {json.dumps(item, ensure_ascii=True)}\n\n"
                last_count = len(events_payload)
                await asyncio.sleep(0.5)

        return StreamingResponse(stream(), media_type="text/event-stream")

    return router


async def _command(request: Request, command: str, session_id: str) -> TeachingSessionResponse:
    orchestrator = request.app.state.runtime.teaching
    try:
        if command in {"pause", "stop"}:
            await request.app.state.runtime.audio.cancel()
            await request.app.state.runtime.hardware.cancel()
        if command == "start":
            return _response(await orchestrator.start(session_id))
        if command == "pause":
            return _response(await orchestrator.pause(session_id))
        if command == "resume":
            return _response(await orchestrator.resume(session_id))
        if command == "stop":
            return _response(await orchestrator.stop(session_id))
    except InvalidTransitionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except TeachingError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    raise HTTPException(status_code=400, detail="Unsupported teaching command.")


def _response(session: TeachingSession) -> TeachingSessionResponse:
    return TeachingSessionResponse(session=session.model_dump(mode="json"))
