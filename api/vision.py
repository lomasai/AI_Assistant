"""Vision API router for browser camera frame analysis."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from server.vision import VisionError, vision_service


class VisionAnalyzeRequest(BaseModel):
    """Request model for one browser camera frame."""

    model_config = ConfigDict(extra="forbid")
    image_base64: str
    timestamp: str | None = None
    include_decision: bool = True
    context: dict[str, Any] = Field(default_factory=dict)


class VisionAnalyzeResponse(BaseModel):
    """Response model for vision analysis."""

    model_config = ConfigDict(extra="forbid")
    ok: bool
    timestamp: str
    latency_ms: float
    face: dict[str, Any]
    eyes_attention: dict[str, Any]
    body_posture: dict[str, Any]
    tracking: dict[str, Any]
    health_behavior: dict[str, Any]
    sensors: dict[str, Any]
    decision: dict[str, Any] | None
    overlays: dict[str, Any] = Field(default_factory=dict)


def build_vision_router() -> APIRouter:
    """Build vision endpoints."""
    router = APIRouter(prefix="/vision", tags=["vision"])

    @router.post("/analyze", response_model=VisionAnalyzeResponse, summary="Analyze one camera frame")
    async def analyze_frame(payload: VisionAnalyzeRequest) -> VisionAnalyzeResponse:
        try:
            result = vision_service.analyze(
                image_base64=payload.image_base64,
                timestamp=payload.timestamp,
                include_decision=payload.include_decision,
                context=payload.context,
            )
        except VisionError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return VisionAnalyzeResponse(**result)

    @router.get("/status", summary="Vision service status")
    async def vision_status() -> dict[str, Any]:
        return vision_service.status()

    return router
