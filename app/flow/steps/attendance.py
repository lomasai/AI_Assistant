from __future__ import annotations

from lomas_core.contracts import ATTENDANCE_MARKED, STUDENT_IDENTIFIED, AttendanceMarked

from app.flow.states import StepResult
from app.flow.step import STEPS, BaseStep

RECOGNISED = "recognised"
ROSTER = "roster"


@STEPS.register("attendance")
class Attendance(BaseStep):
    """Who is actually here.

    Recognised faces are the real source. Until the vision pipeline is wired
    in, and in any classroom where recognition is switched off, it falls back
    to the class roster so the lesson still runs with names.
    """

    name = "attendance"

    def enter(self, ctx) -> None:
        ctx.notes["attendance_started"] = ctx.clock.now()
        ctx.notes["identified"] = set()
        self._unsubscribe = ctx.bus.subscribe(STUDENT_IDENTIFIED, self._on_identified(ctx))

    def _on_identified(self, ctx):
        def handler(_event, payload) -> None:
            student_id = getattr(payload, "student_id", None) or payload.get("student_id")
            if student_id:
                ctx.notes["identified"].add(student_id)

        return handler

    def tick(self, ctx, now: float) -> StepResult:
        seen = ctx.notes.get("identified", set())
        if seen and len(seen) >= len(ctx.roster):
            return StepResult.DONE

        # Stop waiting once the window is up. Without this the stage sits out
        # its whole timeout on any robot whose camera is not running.
        waited = now - ctx.notes["attendance_started"]
        if waited >= ctx.cfg.flow.attendance_wait_seconds:
            return StepResult.DONE
        return StepResult.CONTINUE

    def exit(self, ctx) -> None:
        self._unsubscribe()

        seen = ctx.notes.get("identified", set())
        for student in ctx.roster:
            recognised = student["id"] in seen
            if not recognised and not ctx.cfg.flow.attendance_falls_back_to_roster:
                continue

            ctx.present[student["id"]] = student["name"]
            ctx.repo("session").mark_present(ctx.scope, ctx.session_id, student["id"])
            ctx.bus.publish(
                ATTENDANCE_MARKED,
                AttendanceMarked(
                    session_id=ctx.session_id,
                    student_id=student["id"],
                    name=student["name"],
                    source=RECOGNISED if recognised else ROSTER,
                ),
            )
