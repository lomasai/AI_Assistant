from __future__ import annotations

from lomas_core.contracts import LESSON_SEGMENT, LessonSegment

from app.flow.states import StepResult
from app.flow.step import STEPS, BaseStep

FIRST = 0


@STEPS.register("lesson")
class LessonStep(BaseStep):
    """Walks the lesson one segment at a time.

    One idea per screen and per breath. The display text and the spoken text
    are separate fields so the board can show a short line while the robot
    says the longer version.
    """

    name = "lesson"

    def enter(self, ctx) -> None:
        ctx.notes["segment_index"] = FIRST

    def tick(self, ctx, now: float) -> StepResult:
        index = ctx.notes["segment_index"]
        segments = ctx.lesson.segments
        if index >= len(segments):
            return StepResult.DONE

        segment = segments[index]
        ctx.bus.publish(
            LESSON_SEGMENT,
            LessonSegment(
                session_id=ctx.session_id,
                lesson_id=ctx.lesson.id,
                segment_id=segment.id,
                index=index,
                total=len(segments),
                say=segment.say,
                display=segment.display,
            ),
        )
        ctx.say(segment.say)

        ctx.notes["segment_index"] = index + 1
        ctx.notes["covered"] = ctx.notes.get("covered", 0) + 1
        return StepResult.CONTINUE
