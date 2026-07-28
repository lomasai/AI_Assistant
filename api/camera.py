"""Backend camera preview and status endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import Response, JSONResponse, StreamingResponse

from server.runtime import ApplicationRuntime


def build_camera_router() -> APIRouter:
    router = APIRouter(prefix="/camera", tags=["camera"])

    @router.get("/status", summary="Backend camera status")
    async def camera_status(request: Request) -> dict[str, Any]:
        runtime = _runtime(request)
        return runtime.camera_pipeline.status().as_dict()

    @router.get("/stream.mjpg", summary="Backend MJPEG camera preview", response_model=None)
    async def camera_stream(request: Request) -> Response:
        runtime = _runtime(request)
        status = runtime.camera_pipeline.status()
        if status.state in {"disabled", "error"}:
            return JSONResponse(status_code=503, content=status.as_dict())
        return StreamingResponse(
            runtime.camera_pipeline.mjpeg_frames(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @router.get("/events", summary="Backend camera status event stream")
    async def camera_events(request: Request) -> StreamingResponse:
        runtime = _runtime(request)
        return StreamingResponse(runtime.camera_pipeline.status_events(), media_type="text/event-stream")

    return router


def _runtime(request: Request) -> ApplicationRuntime:
    return request.app.state.runtime
