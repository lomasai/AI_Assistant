from __future__ import annotations

from lomas_core.contracts import STUDENT_DISENGAGED, StudentDisengaged

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
    subscribes = [STUDENT_DISENGAGED]

    def handle(self, _event: str, drifting: StudentDisengaged, ctx: AgentContext) -> None:
        name = ctx.student_name
        if not name:
            # An unrecognised face has no name to use, and a nudge addressed
            # to nobody lands on whoever happens to look up.
            self.log.debug("track %s is drifting but unnamed", drifting.track_id)
            return

        line = self.deps.prompts.line(self.settings.prompt, ctx.language, name=name)
        self.say(ctx, line, student_name=name)
