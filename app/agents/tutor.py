from __future__ import annotations

from lomas_core.contracts import (
    QUESTION_ANSWERED,
    QUESTION_ASKED,
    QuestionAnswered,
    QuestionAsked,
)

from app.agents.base import AGENTS, BaseAgent
from app.context.assembler import AgentContext


@AGENTS.register("tutor")
class Tutor(BaseAgent):
    """Answers what a child asks, at the level of the class.

    It sees the segments already taught and nothing further on, so it cannot
    give away an ending the teacher has not reached yet.
    """

    name = "tutor"
    subscribes = [QUESTION_ASKED]

    def handle(self, _event: str, asked: QuestionAsked, ctx: AgentContext) -> None:
        answer = self.ask(
            ctx,
            grade=ctx.grade,
            subject=ctx.subject,
            vocabulary_level=ctx.vocabulary_level,
            language=ctx.language,
            topic=ctx.lesson_title,
            student_name=asked.student_name or ctx.student_name,
            question=asked.text,
        )
        if not answer:
            self.log.warning("no answer for: %s", asked.text)
            return

        self.say(ctx, answer.text, student_name=asked.student_name)
        self.deps.bus.publish(
            QUESTION_ANSWERED,
            QuestionAnswered(
                session_id=ctx.session_id,
                question=asked.text,
                answer=answer.text,
                provider=answer.provider,
                student_id=asked.student_id,
            ),
        )
