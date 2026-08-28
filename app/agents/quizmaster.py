from __future__ import annotations

from lomas_core.contracts import (
    QUIZ_MARKED,
    QUIZ_POSED,
    QUIZ_RECORDED,
    QUIZ_REQUESTED,
    QuizAnswered,
    QuizMarked,
    QuizPosed,
    QuizRequested,
)

from app.agents.base import AGENTS, BaseAgent
from app.context.assembler import AgentContext

MARK = "mark"
CORRECT = "correct"
GENERATED = "generated"


@AGENTS.register("quizmaster")
class Quizmaster(BaseAgent):
    """Marks what a child actually said, and writes a question when asked.

    Multiple choice is marked by the content pack, which is instant and free.
    Only free text reaches a model, and only after the answer is already
    safely recorded - a class cannot wait on a network round trip to move on.
    """

    name = "quizmaster"
    subscribes = [QUIZ_RECORDED, QUIZ_REQUESTED]

    def handle(self, event: str, payload, ctx: AgentContext) -> None:
        if isinstance(payload, QuizRequested):
            self._write_question(payload, ctx)
        elif isinstance(payload, QuizAnswered):
            self._mark(payload, ctx)

    def _mark(self, answered: QuizAnswered, ctx: AgentContext) -> None:
        if answered.correct is not None:
            return

        verdict = self.ask(
            ctx,
            MARK,
            grade=ctx.grade,
            subject=ctx.subject,
            vocabulary_level=ctx.vocabulary_level,
            language=ctx.language,
            lesson=ctx.lesson_text or ctx.lesson_title,
            question=_asked(ctx) or answered.question_id,
            answer=answered.response,
        )
        if not verdict:
            return

        correct = CORRECT in verdict.text.strip().lower()
        self.deps.repos["answer"].mark(
            ctx.scope, ctx.session_id, answered.student_id, answered.question_id, correct
        )
        self.deps.bus.publish(
            QUIZ_MARKED,
            QuizMarked(
                session_id=ctx.session_id,
                question_id=answered.question_id,
                student_id=answered.student_id,
                correct=correct,
                comment=verdict.text.strip(),
            ),
        )

    def _write_question(self, requested: QuizRequested, ctx: AgentContext) -> None:
        written = self.ask(
            ctx,
            grade=ctx.grade,
            subject=ctx.subject,
            vocabulary_level=ctx.vocabulary_level,
            language=ctx.language,
            content=ctx.lesson_text or ctx.lesson_title,
            task=requested.topic or ctx.topic,
        )
        if not written:
            return

        self.deps.bus.publish(
            QUIZ_POSED,
            QuizPosed(
                session_id=ctx.session_id,
                question_id=f"{GENERATED}:{len(ctx.history)}",
                text=written.text.strip(),
            ),
        )


def _asked(ctx: AgentContext) -> str:
    """The question as it was read out. The event carries an id; the words
    are in the history, which is what the assembler is for."""
    posed = [turn.text for turn in ctx.history if turn.event == QUIZ_POSED]
    return posed[-1] if posed else ""
