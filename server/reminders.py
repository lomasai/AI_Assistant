"""Rule-based reminder engine.

Supported reminders:
- Medicine reminder
- Water reminder
- Sitting duration alert
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time, timedelta, timezone
from typing import Any, Literal


ReminderType = Literal["medicine", "water", "sitting"]
Severity = Literal["low", "medium", "high"]


class ReminderError(Exception):
    """Raised when reminder evaluation fails."""


@dataclass(slots=True)
class ReminderConfig:
    """Configuration for reminder thresholds and schedules."""

    medicine_times: tuple[str, ...] = ("09:00", "21:00")
    medicine_grace_minutes: int = 90
    water_gap_minutes: int = 120
    sitting_limit_minutes: int = 60
    cooldown_minutes: int = 20
    timezone_name: str = "UTC"


@dataclass(slots=True)
class ReminderEvent:
    """A reminder to present to user."""

    type: ReminderType
    message: str
    severity: Severity
    created_at_utc: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ReminderState:
    """Persistent state used by rule checks."""

    last_water_intake_at: datetime | None = None
    last_medicine_taken_at: datetime | None = None
    sitting_started_at: datetime | None = None
    current_posture: str = "unknown"


class ReminderEngine:
    """Rule-based reminder system for health behavior prompts."""

    def __init__(self, config: ReminderConfig | None = None, state: ReminderState | None = None) -> None:
        self.config = config or ReminderConfig()
        self.state = state or ReminderState()
        self._last_emitted: dict[ReminderType, datetime] = {}

    def acknowledge_water_intake(self, at: datetime | None = None) -> None:
        """Mark water intake confirmation."""
        self.state.last_water_intake_at = self._ensure_aware(at or datetime.now(timezone.utc))

    def acknowledge_medicine_intake(self, at: datetime | None = None) -> None:
        """Mark medicine intake confirmation."""
        self.state.last_medicine_taken_at = self._ensure_aware(at or datetime.now(timezone.utc))

    def update_posture(self, posture: str, at: datetime | None = None) -> None:
        """Update posture state and sitting timer."""
        now = self._ensure_aware(at or datetime.now(timezone.utc))
        posture_normalized = posture.strip().lower()
        self.state.current_posture = posture_normalized

        if posture_normalized == "sitting":
            if self.state.sitting_started_at is None:
                self.state.sitting_started_at = now
        else:
            self.state.sitting_started_at = None

    def evaluate(self, context: dict[str, Any] | None = None, now: datetime | None = None) -> list[ReminderEvent]:
        """Evaluate reminder rules and return active events."""
        now_ts = self._ensure_aware(now or datetime.now(timezone.utc))
        ctx = context or {}

        events: list[ReminderEvent] = []
        medicine_event = self._check_medicine(now_ts, ctx)
        if medicine_event:
            events.append(medicine_event)

        water_event = self._check_water(now_ts, ctx)
        if water_event:
            events.append(water_event)

        sitting_event = self._check_sitting(now_ts, ctx)
        if sitting_event:
            events.append(sitting_event)

        return events

    def _check_medicine(self, now_ts: datetime, context: dict[str, Any]) -> ReminderEvent | None:
        if bool(context.get("medicine_taken", False)):
            return None

        last_taken = self._parse_datetime(context.get("last_medicine_taken_at")) or self.state.last_medicine_taken_at
        schedule = context.get("medicine_times", self.config.medicine_times)
        if not isinstance(schedule, (list, tuple)):
            raise ReminderError("medicine_times must be a list/tuple of HH:MM strings.")

        grace_minutes = int(context.get("medicine_grace_minutes", self.config.medicine_grace_minutes))
        window = timedelta(minutes=max(1, grace_minutes))

        for schedule_str in schedule:
            due_local = self._today_schedule_datetime(now_ts, str(schedule_str))
            if now_ts < due_local:
                continue

            if now_ts - due_local > window:
                continue

            if last_taken and last_taken >= due_local:
                continue

            reminder_type: ReminderType = "medicine"
            if not self._can_emit(reminder_type, now_ts):
                return None

            self._mark_emitted(reminder_type, now_ts)
            return ReminderEvent(
                type="medicine",
                message="Did you take your medicine?",
                severity="high",
                created_at_utc=now_ts.astimezone(timezone.utc).isoformat(),
                metadata={
                    "due_time": due_local.isoformat(),
                    "grace_minutes": grace_minutes,
                },
            )

        return None

    def _check_water(self, now_ts: datetime, context: dict[str, Any]) -> ReminderEvent | None:
        threshold = int(context.get("water_gap_minutes_threshold", self.config.water_gap_minutes))
        explicit_gap = context.get("water_gap_minutes")

        if explicit_gap is not None:
            try:
                gap_minutes = float(explicit_gap)
            except (TypeError, ValueError):
                raise ReminderError("water_gap_minutes must be numeric.") from None
        else:
            last_water = self._parse_datetime(context.get("last_water_intake_at")) or self.state.last_water_intake_at
            if last_water is None:
                return None
            gap_minutes = (now_ts - last_water).total_seconds() / 60.0

        if gap_minutes < threshold:
            return None

        reminder_type: ReminderType = "water"
        if not self._can_emit(reminder_type, now_ts):
            return None
        self._mark_emitted(reminder_type, now_ts)

        return ReminderEvent(
            type="water",
            message="Please drink water.",
            severity="medium",
            created_at_utc=now_ts.astimezone(timezone.utc).isoformat(),
            metadata={
                "water_gap_minutes": round(gap_minutes, 2),
                "threshold_minutes": threshold,
            },
        )

    def _check_sitting(self, now_ts: datetime, context: dict[str, Any]) -> ReminderEvent | None:
        posture = str(context.get("posture", self.state.current_posture or "unknown")).lower()
        threshold = int(context.get("sitting_limit_minutes", self.config.sitting_limit_minutes))

        explicit_sitting_minutes = context.get("sitting_minutes")
        if explicit_sitting_minutes is not None:
            try:
                sitting_minutes = float(explicit_sitting_minutes)
            except (TypeError, ValueError):
                raise ReminderError("sitting_minutes must be numeric.") from None
        else:
            if posture != "sitting":
                return None
            if self.state.sitting_started_at is None:
                self.state.sitting_started_at = now_ts
                return None
            sitting_minutes = (now_ts - self.state.sitting_started_at).total_seconds() / 60.0

        if posture != "sitting" or sitting_minutes < threshold:
            return None

        reminder_type: ReminderType = "sitting"
        if not self._can_emit(reminder_type, now_ts):
            return None
        self._mark_emitted(reminder_type, now_ts)

        return ReminderEvent(
            type="sitting",
            message="You have been sitting too long. Please take a short walk.",
            severity="medium",
            created_at_utc=now_ts.astimezone(timezone.utc).isoformat(),
            metadata={
                "sitting_minutes": round(sitting_minutes, 2),
                "threshold_minutes": threshold,
            },
        )

    def _can_emit(self, reminder_type: ReminderType, now_ts: datetime) -> bool:
        last = self._last_emitted.get(reminder_type)
        if last is None:
            return True
        cooldown = timedelta(minutes=max(1, self.config.cooldown_minutes))
        return now_ts - last >= cooldown

    def _mark_emitted(self, reminder_type: ReminderType, now_ts: datetime) -> None:
        self._last_emitted[reminder_type] = now_ts

    @staticmethod
    def _ensure_aware(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def _today_schedule_datetime(self, now_ts: datetime, hhmm: str) -> datetime:
        try:
            hour, minute = hhmm.split(":")
            due_time = time(hour=int(hour), minute=int(minute))
        except Exception as exc:  # noqa: BLE001
            raise ReminderError(f"Invalid medicine time format '{hhmm}', expected HH:MM.") from exc
        return now_ts.replace(hour=due_time.hour, minute=due_time.minute, second=0, microsecond=0)

    @staticmethod
    def _parse_datetime(value: Any) -> datetime | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            return ReminderEngine._ensure_aware(value)
        if not isinstance(value, str) or not value.strip():
            return None
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        return ReminderEngine._ensure_aware(parsed)


reminder_engine = ReminderEngine()
