from __future__ import annotations

from lomas_core.contracts import QUIZ_RECORDED, QuizAnswered

# What happens to an answer, wherever it was given. The quiz step asks
# questions at the end of a class and the teach step asks them in the middle
# of one, and a child's answer has to be recorded the same way in both -
# otherwise a report can tell you which step was running when they answered.

ASKED = "questions_asked"   # ids already put to the class, so nothing is asked twice


def record(ctx, answered: QuizAnswered) -> None:
    """One answer, into the append-only record and back onto the bus.

    Announced after the row exists, so whoever marks free text is updating
    something rather than racing the insert.
    """
    ctx.repo("answer").record(
        ctx.scope,
        session_id=ctx.session_id,
        student_id=answered.student_id,
        question_ref=answered.question_id,
        response=answered.response,
        correct=answered.correct,
        latency_ms=answered.latency_ms,
    )
    ctx.bus.publish(QUIZ_RECORDED, answered)


def remember_asked(ctx, question_id: str) -> None:
    ctx.notes.setdefault(ASKED, []).append(question_id)


def still_to_ask(ctx, questions) -> list:
    """The questions nobody has been asked yet.

    A class that has just answered four questions while the lesson was being
    taught should not be asked the same four again at the end of it.
    """
    asked = set(ctx.notes.get(ASKED, ()))
    return [question for question in questions if question.id not in asked]
