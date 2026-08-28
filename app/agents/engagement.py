from __future__ import annotations

from lomas_core.contracts import (
    SESSION_OPENED,
    STUDENT_DISENGAGED,
    TEACHER_NUDGING,
    NudgingSet,
    StudentDisengaged,
)

from app.agents.base import AGENTS, BaseAgent
from app.context.assembler import AgentContext


@AGENTS.register("engagement")
class Engagement(BaseAgent):
    """Invites a drifting child back into the lesson.

    Every nudge is a question. Never a correction, never an observation about
    the child. "Meera, what do you think happens next?" is an invitation;
    "Meera, you seem distracted" is a machine reprimanding a ten year old in
    front of her class, and it is the fastest way to get the robot switched
    off for good. The phrasings live in the prompt file so a teacher's
    feedback is a content edit, and a test checks the banned words.
    """

    name = "engagement"
    subscribes = [STUDENT_DISENGAGED, TEACHER_NUDGING, SESSION_OPENED]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.nudging = True

    def handle(self, event: str, payload, ctx: AgentContext) -> None:
        if isinstance(payload, NudgingSet):
            self.nudging = payload.enabled
            self.log.info("nudging %s", "on" if payload.enabled else "off")
            return

        if event == SESSION_OPENED:
            self.nudging = True  # the switch is per session, never sticky
            return

        if not self.nudging:
            return
        self._invite(payload, ctx)

    def _invite(self, drifting: StudentDisengaged, ctx: AgentContext) -> None:
        name = ctx.student_name
        if not name:
            # An unrecognised face has no name to use, and a nudge addressed
            # to nobody lands on whoever happens to look up.
            self.log.debug("track %s is drifting but unnamed", drifting.track_id)
            return

        line = self.deps.prompts.line(self.settings.prompt, ctx.language, name=name)
        self.say(ctx, line, student_name=name)
