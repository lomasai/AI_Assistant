from __future__ import annotations

from app.flow.states import StepResult
from app.flow.step import STEPS, BaseStep

GREETING = "greeting"


@STEPS.register("greeting")
class Greeting(BaseStep):
    """Welcomes the class as a class.

    It names nobody. A roll of names at the door singles a few children out
    and leaves the rest waiting; names are held back for when the robot asks
    someone a question directly, where being named means something.
    """

    name = "greeting"

    def enter(self, ctx) -> None:
        ctx.notes["greeted"] = False

    def tick(self, ctx, now: float) -> StepResult:
        if not ctx.notes["greeted"]:
            ctx.say(
                ctx.notes["prompts"].line(
                    GREETING, ctx.language, klass=ctx.notes.get("class_name", ""),
                )
            )
            ctx.notes["greeted"] = True
        return StepResult.DONE
