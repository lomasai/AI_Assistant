"""Safe hardware-control foundation for simulated and future ESP32 motion."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from server.config import HardwareConfig
from server.interfaces import HardwareAck, HardwareCommand, HardwareController


PROTOCOL_VERSION = "hardware.v1"
SAFE_ACTIONS = {"neutral", "small_nod", "small_head_turn", "reset_position", "stop"}
CommandStatus = Literal["ready", "disabled", "connected", "disconnected", "moving", "safe_stopped", "emergency_stopped"]


class HardwareError(Exception):
    """Raised for controlled hardware command failures."""


class HardwareCommandRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["neutral", "small_nod", "small_head_turn", "reset_position"]
    params: dict[str, float] = Field(default_factory=dict)


class HardwareResetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    confirm: bool = False


class Transport(Protocol):
    async def connect(self) -> None:
        ...

    async def close(self) -> None:
        ...

    async def send(self, payload: dict[str, Any]) -> dict[str, Any]:
        ...

    def health(self) -> dict[str, Any]:
        ...


class SimulatedESP32Transport:
    """Deterministic transport used by tests and mock runtime."""

    def __init__(self) -> None:
        self.connected = False
        self.delay_seconds = 0.0
        self.fail_next = False
        self.drop_next = False
        self.sent: list[dict[str, Any]] = []

    async def connect(self) -> None:
        self.connected = True

    async def close(self) -> None:
        self.connected = False

    async def send(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.connected:
            raise HardwareError("Hardware transport is disconnected.")
        if self.delay_seconds:
            await asyncio.sleep(self.delay_seconds)
        if self.drop_next:
            self.drop_next = False
            await asyncio.sleep(0)
            raise TimeoutError("Hardware acknowledgement timed out.")
        if self.fail_next:
            self.fail_next = False
            return {"ok": False, "status": "failed", "command_id": payload.get("command_id", "")}
        self.sent.append(dict(payload))
        return {"ok": True, "status": "ack", "command_id": payload.get("command_id", "")}

    def health(self) -> dict[str, Any]:
        return {"transport": "mock", "connected": self.connected}


class SerialESP32Transport:
    """Serial adapter placeholder. Physical writes are gated by config."""

    def __init__(self, config: HardwareConfig) -> None:
        self.config = config
        self.connected = False

    async def connect(self) -> None:
        if not self.config.physical_output_enabled:
            raise HardwareError("Physical output is disabled.")
        try:
            __import__("serial")
        except Exception as exc:  # noqa: BLE001
            raise HardwareError("pyserial is required for ESP32 serial transport.") from exc
        self.connected = True

    async def close(self) -> None:
        self.connected = False

    async def send(self, payload: dict[str, Any]) -> dict[str, Any]:
        _ = payload
        raise HardwareError("Serial transport is not activated in Phase 7A.")

    def health(self) -> dict[str, Any]:
        return {"transport": "serial", "connected": self.connected}


@dataclass
class AuditRecord:
    command_id: str
    action: str
    status: str
    timestamp_utc: str
    physical_output_enabled: bool
    detail: str = ""

    def safe_dict(self) -> dict[str, Any]:
        return {
            "command_id": self.command_id,
            "action": self.action,
            "status": self.status,
            "timestamp_utc": self.timestamp_utc,
            "physical_output_enabled": self.physical_output_enabled,
            "detail": self.detail,
        }


@dataclass
class HardwareRuntime(HardwareController):
    config: HardwareConfig
    transport: Transport | None = None
    state: CommandStatus = "disabled"
    emergency_stopped: bool = False
    last_motion_at: float = 0.0
    last_heartbeat_at: float = 0.0
    seen_commands: deque[str] = field(default_factory=deque)
    audit: deque[AuditRecord] = field(default_factory=deque)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def __post_init__(self) -> None:
        if self.transport is None:
            self.transport = SerialESP32Transport(self.config) if self.config.provider == "esp32" else SimulatedESP32Transport()
        self.audit = deque(maxlen=self.config.audit_history_limit)
        self.seen_commands = deque(maxlen=self.config.duplicate_retention)

    async def start(self) -> None:
        async with self._lock:
            if not self.config.enabled or self.config.provider == "disabled":
                self.state = "disabled"
                return
            await self.transport.connect()
            self.state = "connected"
            self.last_heartbeat_at = _now_ts()
            if self.config.neutral_on_startup:
                await self._send_safe(HardwareCommand(str(uuid4()), PROTOCOL_VERSION, _now(), "neutral", {}))

    async def stop(self) -> None:
        async with self._lock:
            if self.config.neutral_on_shutdown and self.state not in {"disabled", "emergency_stopped"}:
                await self._send_safe(HardwareCommand(str(uuid4()), PROTOCOL_VERSION, _now(), "stop", {}))
            await self.transport.close()
            self.state = "safe_stopped" if self.config.enabled else "disabled"

    async def submit(self, command: HardwareCommand) -> HardwareAck:
        async with self._lock:
            try:
                self._validate(command)
                ack = await self._send_with_retries(command)
                self._remember(command.command_id)
                if not self.emergency_stopped:
                    self.state = "connected" if ack.ok else "safe_stopped"
                self._record(command.command_id, command.action, ack.status, ack.detail)
                return ack
            except HardwareError as exc:
                self.state = "safe_stopped" if not self.emergency_stopped else "emergency_stopped"
                self._record(command.command_id, command.action, "rejected", str(exc))
                return _ack(command.command_id, False, "rejected", str(exc))

    async def submit_predefined_action(self, action: str, params: dict[str, float] | None = None) -> HardwareAck:
        command = HardwareCommand(
            command_id=str(uuid4()),
            protocol_version=PROTOCOL_VERSION,
            timestamp_utc=_now(),
            action=action,
            params=params or {},
        )
        return await self.submit(command)

    async def cancel(self) -> HardwareAck:
        async with self._lock:
            if self.emergency_stopped:
                command_id = str(uuid4())
                self.state = "emergency_stopped"
                self._record(command_id, "stop", "emergency_stopped", "emergency_stop_active")
                return _ack(command_id, True, "emergency_stopped", "emergency_stop_active")
            self.state = "safe_stopped"
            command = HardwareCommand(str(uuid4()), PROTOCOL_VERSION, _now(), "stop", {})
            ack = await self._best_effort_safe_command(command, "cancelled")
            self._record(command.command_id, "stop", "cancelled", "motion_cancelled")
            return ack

    async def emergency_stop(self) -> HardwareAck:
        async with self._lock:
            self.emergency_stopped = True
            self.state = "emergency_stopped"
            command = HardwareCommand(str(uuid4()), PROTOCOL_VERSION, _now(), "stop", {})
            await self._best_effort_safe_command(command, "emergency_stopped")
            ack = _ack(command.command_id, True, "emergency_stopped", "latched")
            self._record(command.command_id, "stop", "emergency_stopped", "emergency_stop")
            return ack

    async def reset_emergency_stop(self, *, confirm: bool) -> HardwareAck:
        async with self._lock:
            command_id = str(uuid4())
            if not confirm:
                self._record(command_id, "reset_position", "rejected", "reset_confirmation_required")
                return _ack(command_id, False, "rejected", "reset_confirmation_required")
            self.emergency_stopped = False
            self.state = "safe_stopped" if self.config.enabled else "disabled"
            self._record(command_id, "reset_position", "emergency_stop_reset", "explicit_reset")
            return _ack(command_id, True, "emergency_stop_reset", "explicit_reset")

    async def heartbeat(self) -> HardwareAck:
        async with self._lock:
            command = HardwareCommand(str(uuid4()), PROTOCOL_VERSION, _now(), "stop", {"heartbeat": 1.0})
            ack = await self._send_safe(command)
            self.last_heartbeat_at = _now_ts()
            return ack

    async def mark_connection_lost(self) -> None:
        async with self._lock:
            self.state = "disconnected"
            if self.config.emergency_stop_on_lost_connection:
                self.emergency_stopped = True
                self.state = "emergency_stopped"
            self._record(str(uuid4()), "stop", self.state, "connection_lost")

    def permitted_actions(self) -> list[dict[str, Any]]:
        return [{"action": action, "enabled": action in self.config.permitted_actions} for action in self.config.permitted_actions]

    def history(self) -> list[dict[str, Any]]:
        return [record.safe_dict() for record in self.audit]

    def health(self) -> dict[str, Any]:
        return {
            "enabled": self.config.enabled,
            "provider": self.config.provider,
            "transport": self.config.transport,
            "physical_output_enabled": self.config.physical_output_enabled,
            "hardware_profile_approved": self.config.hardware_profile_approved,
            "state": self.state,
            "emergency_stopped": self.emergency_stopped,
            "permitted_actions": list(self.config.permitted_actions),
            "limits": {
                "servo_min_angle_deg": self.config.servo_limits.min_angle_deg,
                "servo_max_angle_deg": self.config.servo_limits.max_angle_deg,
                "servo_max_speed_deg_per_second": self.config.servo_limits.max_speed_deg_per_second,
                "motor_max_speed_percent": self.config.motor_limits.max_speed_percent,
                "motion_cooldown_seconds": self.config.motion_cooldown_seconds,
                "max_continuous_motion_seconds": self.config.max_continuous_motion_seconds,
            },
            "transport_health": self.transport.health(),
            "hardware_profile_blockers": self.profile_blockers(),
        }

    def profile_blockers(self) -> list[str]:
        if self.config.physical_output_enabled and self.config.hardware_profile_approved:
            return []
        blockers = []
        for label, value in {
            "ESP32 board": self.config.esp32_board,
            "communication method": self.config.communication_method,
            "serial port": self.config.serial_port,
            "servo or motor driver board": self.config.driver_board,
            "power supply": self.config.power_supply,
            "emergency stop method": self.config.emergency_stop_method,
        }.items():
            if not str(value).strip():
                blockers.append(label)
        if not self.config.hardware_profile_approved:
            blockers.append("approved hardware profile")
        if not self.config.physical_output_enabled:
            blockers.append("physical output enable approval")
        return blockers

    def _validate(self, command: HardwareCommand) -> None:
        if not self.config.enabled:
            raise HardwareError("Hardware control is disabled.")
        if self.emergency_stopped:
            raise HardwareError("Emergency stop is active.")
        if command.protocol_version != PROTOCOL_VERSION:
            raise HardwareError("Unsupported command protocol.")
        if command.command_id in self.seen_commands:
            raise HardwareError("Duplicate command rejected.")
        if command.action not in SAFE_ACTIONS or command.action not in set(self.config.permitted_actions) | {"stop"}:
            raise HardwareError("Unsupported hardware action.")
        age = _now_ts() - _parse_ts(command.timestamp_utc)
        if age > self.config.stale_command_seconds:
            raise HardwareError("Stale command rejected.")
        if command.action != "stop" and not self.config.unsafe_operating_zone_clear:
            raise HardwareError("Unsafe operating zone is not confirmed clear.")
        if _now_ts() - self.last_motion_at < self.config.motion_cooldown_seconds and command.action != "stop":
            raise HardwareError("Motion cooldown is active.")
        self._validate_params(command)

    def _validate_params(self, command: HardwareCommand) -> None:
        angle = float(command.params.get("angle_deg", 0.0))
        speed = float(command.params.get("speed_deg_per_second", 0.0))
        duration = float(command.params.get("duration_seconds", 0.0))
        motor_speed = float(command.params.get("motor_speed_percent", 0.0))
        if not (self.config.servo_limits.min_angle_deg <= angle <= self.config.servo_limits.max_angle_deg):
            raise HardwareError("Servo angle is out of range.")
        if speed > self.config.servo_limits.max_speed_deg_per_second:
            raise HardwareError("Servo speed is out of range.")
        if duration > self.config.max_continuous_motion_seconds or duration > self.config.servo_limits.max_duration_seconds:
            raise HardwareError("Motion duration is out of range.")
        if motor_speed > self.config.motor_limits.max_speed_percent:
            raise HardwareError("Motor speed is out of range.")

    async def _send_with_retries(self, command: HardwareCommand) -> HardwareAck:
        attempts = self.config.retry_limit + 1
        last_detail = ""
        for _ in range(attempts):
            try:
                return await asyncio.wait_for(self._send_safe(command), timeout=self.config.command_timeout_seconds)
            except (TimeoutError, asyncio.TimeoutError) as exc:
                last_detail = exc.__class__.__name__
        await self._safe_stop_after_failure()
        return _ack(command.command_id, False, "timeout", last_detail)

    async def _send_safe(self, command: HardwareCommand) -> HardwareAck:
        if not self.config.physical_output_enabled and self.config.provider != "mock":
            return _ack(command.command_id, False, "physical_output_disabled", "physical_output_disabled")
        payload = {
            "version": command.protocol_version,
            "command_id": command.command_id,
            "timestamp_utc": command.timestamp_utc,
            "action": command.action,
            "params": _bounded_params(command.params),
        }
        response = await self.transport.send(payload)
        if command.action != "stop":
            self.last_motion_at = _now_ts()
            self.state = "moving"
        return _ack(
            str(response.get("command_id") or command.command_id),
            bool(response.get("ok")),
            str(response.get("status") or "ack"),
            "",
        )

    async def _best_effort_safe_command(self, command: HardwareCommand, fallback_status: str) -> HardwareAck:
        try:
            return await asyncio.wait_for(self._send_safe(command), timeout=self.config.command_timeout_seconds)
        except Exception as exc:  # noqa: BLE001
            return _ack(command.command_id, False, fallback_status, exc.__class__.__name__)

    async def _safe_stop_after_failure(self) -> None:
        self.state = "safe_stopped"
        if self.config.emergency_stop_on_lost_connection:
            self.emergency_stopped = True
            self.state = "emergency_stopped"

    def _remember(self, command_id: str) -> None:
        self.seen_commands.append(command_id)

    def _record(self, command_id: str, action: str, status: str, detail: str = "") -> None:
        self.audit.append(
            AuditRecord(
                command_id=command_id,
                action=action,
                status=status,
                timestamp_utc=_now(),
                physical_output_enabled=self.config.physical_output_enabled,
                detail=detail,
            )
        )


def build_hardware_runtime(config: HardwareConfig) -> HardwareRuntime:
    return HardwareRuntime(config=config)


def command_for_action(action: str, config: HardwareConfig) -> HardwareCommand:
    params = {
        "neutral": {"angle_deg": 0.0, "speed_deg_per_second": min(10.0, config.servo_limits.max_speed_deg_per_second), "duration_seconds": 0.1},
        "reset_position": {"angle_deg": 0.0, "speed_deg_per_second": min(10.0, config.servo_limits.max_speed_deg_per_second), "duration_seconds": 0.2},
        "small_nod": {"angle_deg": min(10.0, config.servo_limits.max_angle_deg), "speed_deg_per_second": min(20.0, config.servo_limits.max_speed_deg_per_second), "duration_seconds": 0.4},
        "small_head_turn": {"angle_deg": min(12.0, config.servo_limits.max_angle_deg), "speed_deg_per_second": min(20.0, config.servo_limits.max_speed_deg_per_second), "duration_seconds": 0.4},
    }.get(action)
    if params is None:
        raise HardwareError("Unsupported hardware action.")
    return HardwareCommand(str(uuid4()), PROTOCOL_VERSION, _now(), action, params)


def _ack(command_id: str, ok: bool, status: str, detail: str = "") -> HardwareAck:
    return HardwareAck(command_id=command_id, ok=ok, status=status, detail=detail, timestamp_utc=_now())


def _bounded_params(params: dict[str, Any]) -> dict[str, float]:
    safe: dict[str, float] = {}
    for key in {"angle_deg", "speed_deg_per_second", "duration_seconds", "motor_speed_percent"}:
        if key in params:
            safe[key] = float(params[key])
    return safe


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _now_ts() -> float:
    return datetime.now(timezone.utc).timestamp()


def _parse_ts(value: str) -> float:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError as exc:
        raise HardwareError("Invalid command timestamp.") from exc
