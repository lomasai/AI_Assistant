from __future__ import annotations

from lomas_core.contracts import (
    QUESTION_ANSWERED,
    QUESTION_ASKED,
    QuestionAnswered,
    QuestionAsked,
)

from app.flow.states import StepResult
from app.flow.step import STEPS, BaseStep

TUTOR = "tutor"
NONE_YET = 0


@STEPS.register("interaction")
class Interaction(BaseStep):
    """Open question time.

    Questions arrive as events, from the teacher dashboard or from speech.
    Attribution comes with the event rather than being guessed here - working
    out which child spoke is a vision and dashboard problem, not this step's.
    """

    name = "interaction"

    def enter(self, ctx) -> None:
        ctx.notes["answered"] = NONE_YET
        self._unsubscribe = ctx.bus.subscribe(QUESTION_ASKED, self._on_question(ctx))

    def _on_question(self, ctx):
        def handler(_event, asked: QuestionAsked) -> None:
            answer = self._answer(ctx, asked)
            if not answer:
                return

            ctx.say(answer.text, student_name=asked.student_name, reason=TUTOR)
            ctx.bus.publish(
                QUESTION_ANSWERED,
                QuestionAnswered(
                    session_id=ctx.session_id,
                    question=asked.text,
                    answer=answer.text,
                    provider=answer.provider,
                    student_id=asked.student_id,
                ),
            )
            ctx.notes["answered"] += 1

        return handler

    def _answer(self, ctx, asked: QuestionAsked):
        prompts = ctx.notes["prompts"]
        content = ctx.cfg.content
        messages = prompts.messages(
            TUTOR,
            ctx.language,
            grade=content.grade,
            subject=content.subject,
            vocabulary_level=content.vocabulary_level,
            language=ctx.language,
            topic=ctx.lesson.title,
            student_name=asked.student_name or "",
            question=asked.text,
        )
        return ctx.notes["llm"].complete(messages, language=ctx.language)

    def tick(self, ctx, now: float) -> StepResult:
        if ctx.notes["answered"] >= ctx.cfg.flow.questions_per_lesson:
            return StepResult.DONE
        return StepResult.CONTINUE

    def exit(self, ctx) -> None:
        self._unsubscribe()
