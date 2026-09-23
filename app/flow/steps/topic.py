from __future__ import annotations

from lomas_core.contracts import (
    LESSON_CHANGED,
    TOPIC_CHOSEN,
    TOPIC_REQUESTED,
    LessonChanged,
    TopicChosen,
)

from app.flow.states import StepResult
from app.flow.step import STEPS, BaseStep

ASK = "ask_topic"
CHOSEN = "topic_chosen"
ASKED_FOR = "topic_asked_for"
GAVE_UP = "topic_gave_up"


@STEPS.register("topic")
class Topic(BaseStep):
    """Asks the class what they would like to learn, and waits for an answer.

    The greeting has asked "what shall we learn about today?" since the first
    week, and nothing was listening. This is the part that listens: a child
    says a subject, a lesson is written for it, and the rest of the class
    runs on that lesson exactly as it would on a reviewed one.

    Removing this step from flow.sequence gives back a robot that teaches the
    topic it was started with, which is what a school with a syllabus wants.
    """

    name = "topic"

    def enter(self, ctx) -> None:
        ctx.notes[CHOSEN] = None
        self._unsubscribe = ctx.bus.subscribe(TOPIC_CHOSEN, self._on_chosen(ctx))

        if ctx.notes.get(ASKED_FOR):
            return

        ctx.notes["topic_asked_at"] = ctx.clock.now()
        ctx.say(ctx.notes["prompts"].line(ASK, ctx.language))
        # After the question has been spoken, not before: the microphone
        # opens when the room has heard what it is being asked.
        ctx.bus.publish(TOPIC_REQUESTED, TopicChosen(session_id=ctx.session_id, text=""))

    def _on_chosen(self, ctx):
        def handler(_event, chosen: TopicChosen) -> None:
            if chosen.text.strip():
                ctx.notes[CHOSEN] = chosen.text.strip()
                return
            # An empty one is whoever was listening saying it gave up. The
            # class should not then sit through the rest of the wait in
            # silence for a question already answered.
            ctx.notes[GAVE_UP] = True

        return handler

    def tick(self, ctx, now: float) -> StepResult:
        if ctx.notes.get(ASKED_FOR):
            # The teacher already typed one. Asking anyway would be a robot
            # that does not listen to the person who set it up.
            return StepResult.DONE

        if ctx.notes.get(CHOSEN) or ctx.notes.get(GAVE_UP):
            return StepResult.DONE

        waiting = now - ctx.notes.get("topic_asked_at", now)
        return StepResult.DONE if waiting >= ctx.cfg.flow.topic_wait_seconds else StepResult.CONTINUE

    def exit(self, ctx) -> None:
        self._unsubscribe()

        wanted = ctx.notes.get(CHOSEN)
        author = ctx.notes.get("author")
        if not wanted or author is None:
            return

        try:
            written = author.cached(wanted, ctx.language) or author.write(wanted, ctx.language)
        except Exception as exc:
            # A lesson that cannot be written is the lesson the robot already
            # had, in front of a class that is already sitting down.
            ctx.notes["topic_error"] = str(exc)
            return

        lesson, quiz = written
        ctx.content.add(lesson, quiz)
        if ctx.library is not None:
            # The agents reload the pack from disk mid-lesson; without this
            # they would ask for a lesson nobody has written down.
            ctx.library.remember(lesson, quiz, ctx.language)
        ctx.lesson = lesson
        ctx.topic = lesson.id

        # The session row, not only this object. Every agent reads what is
        # being taught back from that row: on the Pi a class was taught the
        # water cycle while the tutor and the quiz still believed it was
        # about leaves, and deflected every question as off-topic.
        ctx.repo("session").retopic(ctx.scope, ctx.session_id, lesson.id)
        ctx.bus.publish(
            LESSON_CHANGED,
            LessonChanged(session_id=ctx.session_id, lesson_id=lesson.id,
                          title=lesson.title, written=lesson.written),
        )
