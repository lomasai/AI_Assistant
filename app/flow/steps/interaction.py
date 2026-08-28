from __future__ import annotations

from lomas_core.contracts import QUESTION_ANSWERED

from app.flow.states import StepResult
from app.flow.step import STEPS, BaseStep

NONE_YET = 0


@STEPS.register("interaction")
class Interaction(BaseStep):
    """Open question time.

    The step holds the floor open and counts; answering is the tutor agent's
    job. That split is the point - switch the tutor off and the class still
    reaches the quiz, it simply takes no questions.

    Questions arrive as events, from the teacher dashboard or from speech.
    Attribution comes with the event rather than being guessed here: working
    out which child spoke is a vision and dashboard problem, not this step's.
    """

    name = "interaction"

    def enter(self, ctx) -> None:
        ctx.notes["answered"] = NONE_YET
        self._unsubscribe = ctx.bus.subscribe(QUESTION_ANSWERED, self._on_answer(ctx))

    def _on_answer(self, ctx):
        def handler(_event, _answered) -> None:
            ctx.notes["answered"] += 1

        return handler

    def tick(self, ctx, now: float) -> StepResult:
        if ctx.notes["answered"] >= ctx.cfg.flow.questions_per_lesson:
            return StepResult.DONE
        return StepResult.CONTINUE

    def exit(self, ctx) -> None:
        self._unsubscribe()
