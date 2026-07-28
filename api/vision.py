"""Vision API router for browser camera frame analysis."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from server.vision import VisionError, vision_service


class VisionAnalyzeRequest(BaseModel):
    """Request model for one browser camera frame."""

    model_config = ConfigDict(extra="forbid")
    image_base64: str
    timestamp: str | None = None
    include_decision: bool = True
    context: dict[str, Any] = Field(default_factory=dict)


class VisionTrackRequest(BaseModel):
    """Lenient request model for tracking clients with different frame field names."""

    model_config = ConfigDict(extra="allow")
    image_base64: str | None = None
    timestamp: str | None = None
    context: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def normalize_frame_payload(cls, data: Any) -> Any:
        if isinstance(data, str):
            return {"image_base64": data}
        if not isinstance(data, dict):
            return data

        normalized = dict(data)
        if normalized.get("image_base64"):
            return normalized

        for key in (
            "imageBase64",
            "frame_base64",
            "frameBase64",
            "frame",
            "image",
            "image_data",
            "imageData",
            "frame_data",
            "frameData",
            "data_url",
            "dataUrl",
        ):
            value = normalized.get(key)
            if isinstance(value, str) and value.strip():
                normalized["image_base64"] = value
                break
        return normalized


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


class VisionTrackResponse(BaseModel):
    """Response model for tracking one camera frame."""

    model_config = ConfigDict(extra="forbid")
    ok: bool
    timestamp: str
    latency_ms: float
    face: dict[str, Any]
    tracking: dict[str, Any]
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

    @router.post("/track", response_model=VisionTrackResponse, summary="Track face position in one camera frame")
    async def track_frame(payload_body: Any = Body(default=None)) -> VisionTrackResponse:
        try:
            payload = VisionTrackRequest.model_validate(payload_body or {})
        except ValidationError as exc:
            raise HTTPException(status_code=400, detail=exc.errors()) from exc
        if not payload.image_base64:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Frame image is required. Send image_base64, imageBase64, "
                    "frame_base64, frame, image, data_url, or dataUrl."
                ),
            )
        try:
            result = vision_service.analyze(
                image_base64=payload.image_base64,
                timestamp=payload.timestamp,
                include_decision=False,
                context=payload.context,
            )
        except VisionError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return VisionTrackResponse(
            ok=result["ok"],
            timestamp=result["timestamp"],
            latency_ms=result["latency_ms"],
            face=result["face"],
            tracking=result["tracking"],
            overlays=result.get("overlays", {}),
        )

    @router.get("/status", summary="Vision service status")
    async def vision_status() -> dict[str, Any]:
        return vision_service.status()

    return router
