from __future__ import annotations

from lomas_core.contracts import QUIZ_ANSWERED, QUIZ_POSED, QuizAnswered, QuizPosed

from app.flow.states import StepResult
from app.flow.step import STEPS, BaseStep

FIRST = 0


@STEPS.register("quiz")
class QuizStep(BaseStep):
    """Poses questions and records what each child answered.

    Scores are per student, never per class. Answers arrive as events, with
    the teacher attributing whoever spoke.
    """

    name = "quiz"

    def enter(self, ctx) -> None:
        quiz = ctx.content.quiz_for(ctx.lesson.id)
        ctx.notes["quiz"] = quiz
        ctx.notes["quiz_index"] = FIRST
        ctx.notes["quiz_posed"] = None
        ctx.notes["quiz_recorded"] = FIRST
        ctx.notes["quiz_unanswered"] = FIRST
        ctx.notes["quiz_posed_at"] = 0.0
        self._unsubscribe = ctx.bus.subscribe(QUIZ_ANSWERED, self._on_answer(ctx))

    def _on_answer(self, ctx):
        def handler(_event, answered: QuizAnswered) -> None:
            ctx.repo("answer").record(
                ctx.scope,
                session_id=ctx.session_id,
                student_id=answered.student_id,
                question_ref=answered.question_id,
                response=answered.response,
                correct=answered.correct,
                latency_ms=answered.latency_ms,
            )
            ctx.notes["quiz_recorded"] += 1
            ctx.notes["quiz_posed"] = None

        return handler

    def tick(self, ctx, now: float) -> StepResult:
        quiz = ctx.notes["quiz"]
        if quiz is None:
            return StepResult.DONE

        index = ctx.notes["quiz_index"]
        asked_so_far = min(len(quiz.questions), ctx.cfg.flow.quiz_length)
        if index >= asked_so_far:
            return StepResult.DONE

        if ctx.notes["quiz_posed"] is not None:
            # A class where nobody answers still has to reach the end of the
            # lesson, so give up on this question and ask the next one.
            waiting = now - ctx.notes["quiz_posed_at"]
            if waiting < ctx.cfg.flow.answer_wait_seconds:
                return StepResult.CONTINUE
            ctx.notes["quiz_posed"] = None
            ctx.notes["quiz_unanswered"] += 1

        question = quiz.questions[index]
        ctx.notes["quiz_posed"] = question.id
        ctx.notes["quiz_posed_at"] = now
        ctx.notes["quiz_index"] = index + 1

        ctx.bus.publish(
            QUIZ_POSED,
            QuizPosed(
                session_id=ctx.session_id,
                question_id=question.id,
                text=question.ask,
                options=question.options,
            ),
        )
        ctx.say(question.ask)
        return StepResult.CONTINUE

    def exit(self, ctx) -> None:
        self._unsubscribe()
