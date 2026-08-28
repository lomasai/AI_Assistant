from __future__ import annotations

from lomas_core.contracts import ROBOT_STATE, STORY_REQUESTED, StoryRequested

from app.agents.base import AGENTS, BaseAgent
from app.context.assembler import AgentContext

TELLING = "storytelling"
IDLE = "idle"


@AGENTS.register("narrator")
class Narrator(BaseAgent):
    """Tells a short story, and says so.

    The state events bracket the speech so the face screen and the servos can
    change posture for a story without either of them parsing what was said.
    """

    name = "narrator"
    subscribes = [STORY_REQUESTED]

    def handle(self, _event: str, requested: StoryRequested, ctx: AgentContext) -> None:
        story = self.ask(
            ctx,
            grade=ctx.grade,
            subject=ctx.subject,
            vocabulary_level=ctx.vocabulary_level,
            language=ctx.language,
            topic=requested.topic or ctx.topic,
        )
        if not story:
            return

        self._state(ctx, TELLING)
        try:
            self.say(ctx, story.text)
        finally:
            # In a finally block on purpose. A robot stuck in a storytelling
            # pose because the speaker died is a robot a teacher has to reboot.
            self._state(ctx, IDLE)

    def _state(self, ctx: AgentContext, state: str) -> None:
        self.deps.bus.publish(
            ROBOT_STATE, {"session_id": ctx.session_id, "state": state, "by": self.name}
        )
