"""Action execution system for server-side tool actions."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

from server.memory.vector_db import MemoryError, memory_service
from server.reminders import reminder_engine


class ActionExecutionError(Exception):
    """Raised when action execution fails."""


ActionHandler = Callable[[dict[str, Any], dict[str, Any]], Awaitable[dict[str, Any]]]


@dataclass(slots=True)
class ActionResult:
    """Result payload for an executed action."""

    ok: bool
    name: str
    message: str
    data: dict[str, Any] = field(default_factory=dict)
    executed_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def as_dict(self) -> dict[str, Any]:
        """Return serializable action result."""
        return {
            "ok": self.ok,
            "name": self.name,
            "message": self.message,
            "data": self.data,
            "executed_at_utc": self.executed_at_utc,
        }


class ActionExecutionEngine:
    """Executes supported server actions."""

    def __init__(self) -> None:
        self._handlers: dict[str, ActionHandler] = {
            "mark_medicine_taken": self._handle_mark_medicine_taken,
            "log_event": self._handle_log_event,
            "trigger_alert": self._handle_trigger_alert,
            "set_reminder": self._handle_set_reminder,
            "start_tracking": self._handle_start_tracking,
            "stop_tracking": self._handle_stop_tracking,
        }

    async def execute(self, action: dict[str, Any], context: dict[str, Any] | None = None) -> ActionResult:
        """Execute one action dict with shape {'name': str, 'args': object}."""
        ctx = context or {}
        name = str(action.get("name", "")).strip()
        if not name:
            raise ActionExecutionError("Action name is required.")

        args = action.get("args", {})
        if args is None:
            args = {}
        if not isinstance(args, dict):
            raise ActionExecutionError("Action args must be an object.")

        handler = self._handlers.get(name)
        if handler is None:
            return ActionResult(
                ok=False,
                name=name,
                message=f"Unsupported action: {name}",
                data={"supported_actions": sorted(self._handlers)},
            )

        try:
            payload = await handler(args, ctx)
        except Exception as exc:  # noqa: BLE001
            return ActionResult(
                ok=False,
                name=name,
                message=f"Action execution failed: {exc}",
                data={"error_type": exc.__class__.__name__},
            )

        return ActionResult(
            ok=True,
            name=name,
            message="Action executed successfully.",
            data=payload,
        )

    async def _handle_mark_medicine_taken(self, args: dict[str, Any], _: dict[str, Any]) -> dict[str, Any]:
        at_iso = args.get("taken_at")
        if isinstance(at_iso, str) and at_iso.strip():
            try:
                taken_at = datetime.fromisoformat(at_iso.replace("Z", "+00:00"))
            except ValueError as exc:
                raise ActionExecutionError("taken_at must be ISO datetime.") from exc
            reminder_engine.acknowledge_medicine_intake(taken_at)
        else:
            reminder_engine.acknowledge_medicine_intake()

        await self._safe_log_event(
            event_name="medicine_taken",
            metadata={"source": "action_engine", "args": args},
        )
        return {"medicine_taken": True}

    async def _handle_log_event(self, args: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        event_name = str(args.get("event", "")).strip()
        if not event_name:
            raise ActionExecutionError("log_event requires args.event.")

        role = str(args.get("role", "system")).strip().lower()
        message = str(args.get("message", event_name)).strip()
        metadata = args.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ActionExecutionError("metadata must be an object.")

        metadata = {
            **metadata,
            "source": "action_engine",
            "context_keys": sorted(list(context.keys())),
        }

        try:
            log_id = await memory_service.store_conversation_log(role=role, message=message, metadata=metadata)
        except MemoryError as exc:
            raise ActionExecutionError(str(exc)) from exc

        return {"log_id": log_id, "event": event_name}

    async def _handle_trigger_alert(self, args: dict[str, Any], _: dict[str, Any]) -> dict[str, Any]:
        message = str(args.get("message", "")).strip()
        if not message:
            raise ActionExecutionError("trigger_alert requires args.message.")

        severity = str(args.get("severity", "medium")).strip().lower()
        if severity not in {"low", "medium", "high"}:
            raise ActionExecutionError("severity must be one of: low, medium, high.")

        await self._safe_log_event(
            event_name="alert_triggered",
            metadata={"message": message, "severity": severity, "source": "action_engine"},
        )
        return {
            "alert": {"message": message, "severity": severity},
            "dispatch": "queued",
        }

    async def _handle_set_reminder(self, args: dict[str, Any], _: dict[str, Any]) -> dict[str, Any]:
        message = str(args.get("message", "")).strip()
        reminder_time = str(args.get("time", "")).strip()
        if not message:
            raise ActionExecutionError("set_reminder requires args.message.")

        await self._safe_log_event(
            event_name="reminder_set",
            metadata={"message": message, "time": reminder_time or None, "source": "action_engine"},
        )
        return {"reminder": {"message": message, "time": reminder_time or None}, "status": "scheduled"}

    async def _handle_start_tracking(self, args: dict[str, Any], _: dict[str, Any]) -> dict[str, Any]:
        await self._safe_log_event(event_name="tracking_started", metadata={"args": args, "source": "action_engine"})
        return {"tracking": "started"}

    async def _handle_stop_tracking(self, args: dict[str, Any], _: dict[str, Any]) -> dict[str, Any]:
        await self._safe_log_event(event_name="tracking_stopped", metadata={"args": args, "source": "action_engine"})
        return {"tracking": "stopped"}

    async def _safe_log_event(self, event_name: str, metadata: dict[str, Any]) -> None:
        try:
            await memory_service.store_conversation_log(
                role="system",
                message=f"EVENT::{event_name}",
                metadata=metadata,
            )
        except Exception:  # noqa: BLE001
            # Logging failures must not block main action path.
            return


action_engine = ActionExecutionEngine()
