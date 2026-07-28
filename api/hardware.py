"""Protected hardware-control APIs for Phase 7A simulated motion."""

from __future__ import annotations

import hmac
import os
from typing import Annotated, Any

from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import BaseModel, ConfigDict

from server.hardware import HardwareCommandRequest, HardwareError, HardwareResetRequest, command_for_action
from server.runtime import ApplicationRuntime


class CancelResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ack: dict[str, Any]


def build_hardware_router() -> APIRouter:
    router = APIRouter(prefix="/hardware", tags=["hardware"])

    def require_admin(x_admin_token: Annotated[str | None, Header()] = None) -> None:
        expected = os.getenv("ADMIN_API_TOKEN", "").strip()
        if expected and not hmac.compare_digest(x_admin_token or "", expected):
            raise HTTPException(status_code=403, detail="Admin authorization required.")

    def runtime_from(request: Request) -> ApplicationRuntime:
        runtime: ApplicationRuntime | None = getattr(request.app.state, "runtime", None)
        if runtime is None:
            raise HTTPException(status_code=503, detail="Runtime is not ready.")
        return runtime

    @router.get("/health")
    async def health(request: Request, x_admin_token: Annotated[str | None, Header()] = None) -> dict[str, Any]:
        require_admin(x_admin_token)
        return runtime_from(request).hardware.health()

    @router.get("/actions")
    async def actions(request: Request, x_admin_token: Annotated[str | None, Header()] = None) -> dict[str, Any]:
        require_admin(x_admin_token)
        return {"actions": runtime_from(request).hardware.permitted_actions()}

    @router.post("/actions")
    async def submit_action(
        payload: HardwareCommandRequest,
        request: Request,
        x_admin_token: Annotated[str | None, Header()] = None,
    ) -> dict[str, Any]:
        require_admin(x_admin_token)
        runtime = runtime_from(request)
        try:
            command = command_for_action(payload.action, runtime.config.hardware)
        except HardwareError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        command.params.update(payload.params)
        ack = await runtime.hardware.submit(command)
        if not ack.ok:
            raise HTTPException(status_code=409, detail=ack.status)
        return {"ack": _ack_dict(ack)}

    @router.post("/cancel")
    async def cancel(request: Request, x_admin_token: Annotated[str | None, Header()] = None) -> dict[str, Any]:
        require_admin(x_admin_token)
        return {"ack": _ack_dict(await runtime_from(request).hardware.cancel())}

    @router.post("/emergency-stop")
    async def emergency_stop(request: Request, x_admin_token: Annotated[str | None, Header()] = None) -> dict[str, Any]:
        require_admin(x_admin_token)
        return {"ack": _ack_dict(await runtime_from(request).hardware.emergency_stop())}

    @router.post("/emergency-stop/reset")
    async def reset_emergency_stop(
        payload: HardwareResetRequest,
        request: Request,
        x_admin_token: Annotated[str | None, Header()] = None,
    ) -> dict[str, Any]:
        require_admin(x_admin_token)
        return {"ack": _ack_dict(await runtime_from(request).hardware.reset_emergency_stop(confirm=payload.confirm))}

    @router.get("/history")
    async def history(request: Request, x_admin_token: Annotated[str | None, Header()] = None) -> dict[str, Any]:
        require_admin(x_admin_token)
        return {"history": runtime_from(request).hardware.history()}

    return router


def _ack_dict(ack: Any) -> dict[str, Any]:
    return {
        "command_id": ack.command_id,
        "ok": ack.ok,
        "status": ack.status,
        "timestamp_utc": ack.timestamp_utc,
    }
