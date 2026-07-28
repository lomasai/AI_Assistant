"""Versioned audio and voice-turn endpoints."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from server.runtime import ApplicationRuntime


class PushToTalkRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str | None = None


class TTSStartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=1200)


class VoiceAnswerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    transcript: str = Field(min_length=1, max_length=1000)


def build_audio_router() -> APIRouter:
    router = APIRouter(prefix="/audio", tags=["audio"])

    def runtime_from(request: Request) -> ApplicationRuntime:
        runtime: ApplicationRuntime | None = getattr(request.app.state, "runtime", None)
        if runtime is None:
            raise HTTPException(status_code=503, detail="Runtime is not ready.")
        return runtime

    @router.get("/health")
    async def health(request: Request) -> dict[str, Any]:
        runtime = runtime_from(request)
        return {"audio": runtime.audio.status().safe_dict()}

    @router.get("/state")
    async def state(request: Request) -> dict[str, Any]:
        runtime = runtime_from(request)
        return runtime.audio.status().safe_dict()

    @router.post("/push-to-talk/start")
    async def push_to_talk(payload: PushToTalkRequest, request: Request) -> dict[str, Any]:
        runtime = runtime_from(request)
        if not runtime.config.feature_flags.push_to_talk:
            raise HTTPException(status_code=403, detail="Push-to-talk is disabled.")
        return await runtime.audio.push_to_talk(session_id=payload.session_id)

    @router.post("/push-to-talk/cancel")
    async def cancel_push_to_talk(request: Request) -> dict[str, Any]:
        return await runtime_from(request).audio.cancel()

    @router.post("/wake-word/activate")
    async def wake_word_activation(request: Request) -> dict[str, Any]:
        runtime = runtime_from(request)
        if not runtime.config.feature_flags.wake_word:
            return {"activated": False, "status": "disabled"}
        return await runtime.audio.wait_for_wake_word()

    @router.post("/voice-answer")
    async def submit_voice_answer(payload: VoiceAnswerRequest, request: Request) -> dict[str, Any]:
        runtime = runtime_from(request)
        return await runtime.audio.submit_voice_answer(payload.session_id, payload.transcript)

    @router.post("/tts/start")
    async def tts_start(payload: TTSStartRequest, request: Request) -> dict[str, Any]:
        runtime = runtime_from(request)
        if not runtime.config.feature_flags.tts_output:
            raise HTTPException(status_code=403, detail="TTS output is disabled.")
        return await runtime.audio.speak(payload.text)

    @router.post("/tts/cancel")
    async def tts_cancel(request: Request) -> dict[str, Any]:
        return await runtime_from(request).audio.cancel()

    @router.get("/events")
    async def events(request: Request) -> StreamingResponse:
        async def stream():
            last_count = 0
            while True:
                snapshot = runtime_from(request).audio.event_snapshot()
                for item in snapshot[last_count:]:
                    yield f"event: audio\ndata: {json.dumps(item, ensure_ascii=True)}\n\n"
                last_count = len(snapshot)
                await asyncio.sleep(0.5)

        return StreamingResponse(stream(), media_type="text/event-stream")

    return router

