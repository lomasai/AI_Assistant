from __future__ import annotations

from lomas_core.contracts import (
    HAND_UP,
    LESSON_SEGMENT,
    QUESTION_ANSWERED,
    QUESTION_ASKED,
    QUIZ_ANSWERED,
    QUIZ_MARKED,
    QUIZ_POSED,
    ROBOT_STATE,
    UNDERSTANDING_CHECKED,
    LessonSegment,
    QuizAnswered,
    QuizPosed,
    UnderstandingChecked,
)

from app.flow.states import StepResult
from app.flow.step import STEPS, BaseStep
from app.flow.steps import marks

FIRST = 0

# Where the step is in one idea's life.
SAYING = "saying"      # the next segment is due
SETTLING = "settling"  # said, and the room has a moment to speak
CHECKING = "checking"  # a question is out there, or doubts were invited

DOUBTS = "doubts"
QUESTION = "question"
ALTERNATE = "alternate"

DOUBTS_LINE = "check_doubts"
ASK_LINE = "check_ask"
RESUME_LINE = "check_resume"
NOBODY_LINE = "check_nobody"

INDEX = "segment_index"
PHASE = "teach_phase"
UNTIL = "teach_until"
POSED = "teach_posed"
CHECKS = "teach_checks"
NEXT_CHILD = "teach_next_child"
RESUME = "teach_resume"
HOLDING = "teach_holding"
MIC_OPEN = "teach_mic_open"
LISTENING = "listening"


@STEPS.register("teach")
class Teach(BaseStep):
    """One idea, then the class gets it back.

    The old way was three steps in a row: read the whole lesson out, then
    take questions, then run the quiz. A class sat through six paragraphs
    before anybody was asked anything, and by then the first idea was ten
    minutes old. This is the same three things interleaved - say a segment,
    hand it back, answer whatever comes, carry on - which is what a teacher
    does and what "did you understand?" is for.

    It publishes the same events the three steps do, so reports, the board
    and the marking agent need to know nothing about it. Take it out of
    flow.sequence and put [lesson, interaction, quiz] back, and the robot
    reads the lesson straight through again.
    """

    name = "teach"

    def enter(self, ctx) -> None:
        ctx.notes[INDEX] = FIRST
        ctx.notes[PHASE] = SAYING
        ctx.notes[UNTIL] = 0.0
        ctx.notes[POSED] = None
        ctx.notes[CHECKS] = FIRST
        ctx.notes[NEXT_CHILD] = FIRST
        ctx.notes[RESUME] = False
        ctx.notes[HOLDING] = 0.0
        ctx.notes[MIC_OPEN] = False
        self._unsubscribe = [
            # A hand goes up before a word is said. Holding from here means
            # the next idea is not already being spoken when the child
            # starts talking.
            ctx.bus.subscribe(HAND_UP, self._on_question(ctx)),
            ctx.bus.subscribe(QUESTION_ASKED, self._on_question(ctx)),
            ctx.bus.subscribe(QUESTION_ANSWERED, self._on_answered(ctx)),
            ctx.bus.subscribe(QUIZ_ANSWERED, self._on_quiz_answer(ctx)),
            ctx.bus.subscribe(QUIZ_MARKED, self._on_marked(ctx)),
            ctx.bus.subscribe(ROBOT_STATE, self._on_state(ctx)),
        ]

    def exit(self, ctx) -> None:
        for unsubscribe in self._unsubscribe:
            unsubscribe()

    # --- what the room is doing -------------------------------------------

    def _on_question(self, ctx):
        def handler(_event, _asked) -> None:
            # A child is mid-conversation with the robot. The next idea must
            # not land on top of the answer.
            ctx.notes[HOLDING] = ctx.clock.now()

        return handler

    def _on_answered(self, ctx):
        def handler(_event, _answered) -> None:
            ctx.notes[HOLDING] = 0.0
            ctx.notes[RESUME] = True

        return handler

    def _on_quiz_answer(self, ctx):
        def handler(_event, answered: QuizAnswered) -> None:
            if answered.question_id == ctx.notes[POSED]:
                ctx.notes[POSED] = None
                # Held until the marking agent has said something about it,
                # so praise does not arrive over the next idea.
                ctx.notes[HOLDING] = ctx.clock.now()

            # Last: the marking agent answers this on the spot, and would
            # otherwise release a hold that had not been taken yet.
            marks.record(ctx, answered)

        return handler

    def _on_marked(self, ctx):
        def handler(_event, _marked) -> None:
            ctx.notes[HOLDING] = 0.0

        return handler

    def _on_state(self, ctx):
        def handler(_event, state) -> None:
            heard = state.get("state") if isinstance(state, dict) else getattr(state, "state", "")
            ctx.notes[MIC_OPEN] = heard == LISTENING

        return handler

    def _held(self, ctx, now: float) -> bool:
        """Whether somebody else has the floor.

        Every hold here is bounded. Hands free, the microphone is open
        almost all the time - it opens again half a second after each turn -
        so "wait for the microphone" without a bound is a lesson that never
        reaches its second paragraph.
        """
        if ctx.notes.get(LISTENING):
            # Somebody is deliberately recording an answer to this question.
            return True

        if ctx.notes[MIC_OPEN] and ctx.notes[PHASE] == CHECKING:
            # The robot has just asked something and a child may be part way
            # through answering. Past the window plus this, they are not.
            if now < ctx.notes[UNTIL] + ctx.cfg.flow.teach.microphone_grace_seconds:
                return True

        since = ctx.notes[HOLDING]
        if not since:
            return False
        if now - since < ctx.cfg.flow.teach.answer_hold_seconds:
            return True
        ctx.notes[HOLDING] = 0.0
        return False

    # --- one idea at a time ------------------------------------------------

    def tick(self, ctx, now: float) -> StepResult:
        if self._held(ctx, now):
            return StepResult.CONTINUE

        phase = ctx.notes[PHASE]
        if phase == SAYING:
            return self._say_next(ctx)
        if now < ctx.notes[UNTIL]:
            return StepResult.CONTINUE
        if phase == CHECKING:
            self._nobody_answered(ctx)
        return self._next(ctx)

    def _say_next(self, ctx) -> StepResult:
        index = ctx.notes[INDEX]
        segments = ctx.lesson.segments
        if index >= len(segments):
            return StepResult.DONE

        if ctx.notes[RESUME] and ctx.cfg.flow.teach.say_resuming:
            # A child asked something and got an answer; the lesson picking up
            # again should sound like it, not like a new sentence out of air.
            ctx.notes[RESUME] = False
            self._line(ctx, RESUME_LINE)

        segment = segments[index]
        ctx.bus.publish(
            LESSON_SEGMENT,
            LessonSegment(
                session_id=ctx.session_id,
                lesson_id=ctx.lesson.id,
                segment_id=segment.id,
                index=index,
                total=len(segments),
                say=segment.say,
                display=segment.display,
            ),
        )
        ctx.say(segment.say)
        ctx.notes["covered"] = ctx.notes.get("covered", 0) + 1

        # Timed from when the class has heard it, not from when it was
        # queued: the voice speaks a paragraph long after the publish.
        self._check_or_settle(ctx, index)
        return StepResult.CONTINUE

    def _check_or_settle(self, ctx, index: int) -> None:
        teach = ctx.cfg.flow.teach
        now = ctx.clock.now()
        due = (index + 1) % teach.check_every == FIRST
        if not due:
            ctx.notes[PHASE] = SETTLING
            ctx.notes[UNTIL] = now + teach.gap_seconds
            return

        if self._style(ctx) == QUESTION and self._ask_one(ctx, index):
            ctx.notes[PHASE] = CHECKING
            ctx.notes[UNTIL] = ctx.clock.now() + teach.answer_wait_seconds
            return

        self._invite_doubts(ctx, index)
        ctx.notes[PHASE] = CHECKING
        ctx.notes[UNTIL] = ctx.clock.now() + teach.doubt_wait_seconds

    def _style(self, ctx) -> str:
        """Which kind of check this one is.

        alternate keeps a class awake: a question they have to answer, then
        an invitation to ask one, then a question again.
        """
        wanted = ctx.cfg.flow.teach.check_style
        if wanted != ALTERNATE:
            return wanted
        return QUESTION if ctx.notes[CHECKS] % 2 == FIRST else DOUBTS

    def _next(self, ctx) -> StepResult:
        ctx.notes[INDEX] += 1
        ctx.notes[PHASE] = SAYING
        ctx.notes[POSED] = None
        return StepResult.CONTINUE

    # --- handing the idea back --------------------------------------------

    def _invite_doubts(self, ctx, index: int) -> None:
        ctx.notes[CHECKS] += 1
        said = self._line(ctx, DOUBTS_LINE)
        ctx.bus.publish(
            UNDERSTANDING_CHECKED,
            UnderstandingChecked(session_id=ctx.session_id, index=index, kind=DOUBTS, text=said),
        )

    def _ask_one(self, ctx, index: int) -> bool:
        """A question from the lesson's own quiz, put to one child by name.

        The same question the end-of-class quiz would have asked, asked while
        the idea is still in the room - and then not asked again at the end.
        """
        question = self._question_for(ctx)
        if question is None:
            return False

        ctx.notes[CHECKS] += 1
        ctx.notes[POSED] = question.id
        marks.remember_asked(ctx, question.id)

        child = self._whose_turn(ctx)
        ctx.bus.publish(
            QUIZ_POSED,
            QuizPosed(
                session_id=ctx.session_id,
                question_id=question.id,
                text=question.ask,
                options=question.options,
            ),
        )
        ctx.bus.publish(
            UNDERSTANDING_CHECKED,
            UnderstandingChecked(session_id=ctx.session_id, index=index, kind=QUESTION,
                                 text=question.ask, student_name=child),
        )
        asked = self._line(ctx, ASK_LINE, name=child, question=question.ask) if child else ""
        if not asked:
            ctx.say(question.ask)
        return True

    def _question_for(self, ctx):
        quiz = ctx.content.quiz_for(ctx.lesson.id)
        if quiz is None:
            return None
        left = marks.still_to_ask(ctx, quiz.questions)
        return left[FIRST] if left else None

    def _whose_turn(self, ctx) -> str:
        """Round the roster, so the same confident child is not asked every
        time and the quiet ones are asked at all."""
        if not ctx.cfg.flow.teach.name_a_child or not ctx.roster:
            return ""
        turn = ctx.notes[NEXT_CHILD] % len(ctx.roster)
        ctx.notes[NEXT_CHILD] = turn + 1
        return str(ctx.roster[turn].get("name", ""))

    def _nobody_answered(self, ctx) -> None:
        if ctx.notes[POSED] is None:
            return
        ctx.notes[POSED] = None
        self._line(ctx, NOBODY_LINE)

    def _line(self, ctx, prompt: str, **values) -> str:
        prompts = ctx.notes.get("prompts")
        if prompts is None:
            return ""
        try:
            said = prompts.line(prompt, ctx.language, **values)
        except Exception:  # a missing prompt file must not stop a lesson
            return ""
        ctx.say(said)
        return said
