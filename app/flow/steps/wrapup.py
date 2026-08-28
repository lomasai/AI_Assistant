from __future__ import annotations

from app.flow.states import StepResult
from app.flow.step import STEPS, BaseStep

WRAPUP = "wrapup"
NONE = 0


@STEPS.register("wrapup")
class Wrapup(BaseStep):
    """Closes the lesson.

    It thanks the class, not a list of names, and it says what was covered
    rather than who did well - nothing here should read as a ranking.
    """

    name = "wrapup"

    def tick(self, ctx, now: float) -> StepResult:
        ctx.say(
            ctx.notes["prompts"].line(
                WRAPUP,
                ctx.language,
                topic=ctx.lesson.title,
                covered=ctx.notes.get("covered", NONE),
                total=len(ctx.lesson),
            )
        )
        return StepResult.DONE
