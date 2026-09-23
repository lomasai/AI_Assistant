from __future__ import annotations

from lomas_core.contracts import (
    QUIZ_ANSWERED,
    QUIZ_MARKED,
    QUIZ_POSED,
    QuizAnswered,
    QuizMarked,
    QuizPosed,
)

from app.flow.states import StepResult
from app.flow.step import STEPS, BaseStep
from app.flow.steps import marks

FIRST = 0
# Set by whoever is recording a child, so the quiz does not move on under them.
LISTENING = "listening"


@STEPS.register("quiz")
class QuizStep(BaseStep):
    """Poses questions and records what each child answered.

    Scores are per student, never per class. Answers arrive as events, with
    the teacher attributing whoever spoke.
    """

    name = "quiz"

    def enter(self, ctx) -> None:
        quiz = ctx.content.quiz_for(ctx.lesson.id)
        # Only what the class has not already answered. The teach step asks
        # these in the middle of the lesson, and asking them again at the end
        # is a robot that was not listening the first time.
        ctx.notes["quiz_left"] = marks.still_to_ask(ctx, quiz.questions) if quiz else []
        ctx.notes["quiz"] = quiz
        ctx.notes["quiz_index"] = FIRST
        ctx.notes["quiz_posed"] = None
        ctx.notes["quiz_recorded"] = FIRST
        ctx.notes["quiz_unanswered"] = FIRST
        ctx.notes["quiz_posed_at"] = 0.0
        ctx.notes["quiz_marking"] = None
        ctx.notes["quiz_marking_since"] = 0.0
        self._unsubscribe = [
            ctx.bus.subscribe(QUIZ_ANSWERED, self._on_answer(ctx)),
            ctx.bus.subscribe(QUIZ_MARKED, self._on_marked(ctx)),
        ]

    def _on_answer(self, ctx):
        def handler(_event, answered: QuizAnswered) -> None:
            ctx.notes["quiz_recorded"] += 1
            # Only the question still waiting. A late answer to the last one
            # must not cut short the wait on this one.
            if answered.question_id == ctx.notes["quiz_posed"]:
                ctx.notes["quiz_posed"] = None
                # Held until it is marked, so what the robot says about this
                # answer comes before the next question and not over it.
                ctx.notes["quiz_marking"] = answered.question_id
                ctx.notes["quiz_marking_since"] = ctx.clock.now()

            # Last, because the marking agent answers this on the spot and
            # would otherwise clear a hold that had not been set yet.
            marks.record(ctx, answered)

        return handler

    def _on_marked(self, ctx):
        def handler(_event, marked: QuizMarked) -> None:
            if marked.question_id == ctx.notes["quiz_marking"]:
                ctx.notes["quiz_marking"] = None

        return handler

    def tick(self, ctx, now: float) -> StepResult:
        questions = ctx.notes["quiz_left"]
        if not questions:
            return StepResult.DONE

        index = ctx.notes["quiz_index"]
        asked_so_far = min(len(questions), ctx.cfg.flow.quiz_length)
        if index >= asked_so_far:
            return StepResult.DONE

        if ctx.notes.get(LISTENING):
            # A child mid-answer. On the Pi the next question was asked while
            # the answer to this one was still being recorded.
            return StepResult.CONTINUE

        if ctx.notes["quiz_marking"] is not None:
            if now - ctx.notes["quiz_marking_since"] < ctx.cfg.flow.mark_wait_seconds:
                return StepResult.CONTINUE
            ctx.notes["quiz_marking"] = None

        if ctx.notes["quiz_posed"] is not None:
            # A class where nobody answers still has to reach the end of the
            # lesson, so give up on this question and ask the next one.
            waiting = now - ctx.notes["quiz_posed_at"]
            if waiting < ctx.cfg.flow.answer_wait_seconds:
                return StepResult.CONTINUE
            ctx.notes["quiz_posed"] = None
            ctx.notes["quiz_unanswered"] += 1

        question = questions[index]
        ctx.notes["quiz_posed"] = question.id
        marks.remember_asked(ctx, question.id)
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
        # From when the question has been heard, not when it was queued. On
        # the Pi the first question waited nineteen seconds behind an answer
        # still being read aloud, and the class got a second and a half.
        ctx.notes["quiz_posed_at"] = ctx.clock.now()
        return StepResult.CONTINUE

    def exit(self, ctx) -> None:
        for unsubscribe in self._unsubscribe:
            unsubscribe()
