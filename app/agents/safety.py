from __future__ import annotations

from lomas_core.contracts import ROBOT_BLOCKED, Blocked

from app.agents.base import AGENTS, BaseAgent
from app.context.assembler import AgentContext

TERM = "term"
MODEL = "model"


@AGENTS.register("safety")
class Safety(BaseAgent):
    """The filter on everything the robot says to a child.

    This one is not an event subscriber, and that is deliberate. A filter that
    listens for an event has already lost the race with the speaker; the voice
    asks it before making a sound. Switch it off in `agents.enabled` and the
    voice is handed a filter that allows everything, so the class still runs.

    Two layers. The term list is deterministic, needs no model and no
    internet, so it is the only check a school can count on being there. The
    model check is better and costs a round trip per line, so it is off until
    a school turns it on.
    """

    name = "safety"
    subscribes: list[str] = []

    def approve(self, text: str, language: str, session_id: str = "") -> bool:
        blocked = self._blocked_term(text)
        if blocked:
            self._report(text, f"{TERM}: {blocked}", session_id)
            return False

        if not self.cfg.agents.safety.use_model:
            return True

        if not self._model_allows(text, language):
            self._report(text, MODEL, session_id)
            return False
        return True

    def _blocked_term(self, text: str) -> str:
        lowered = text.lower()
        for term in self.cfg.agents.safety.blocked_terms:
            if term.lower() in lowered:
                return term
        return ""

    def _model_allows(self, text: str, language: str) -> bool:
        rules = self.cfg.agents.safety
        try:
            messages = self.deps.prompts.messages(self.settings.prompt, language, text=text)
            verdict = self.deps.llm.complete(messages, language=language).text.strip().upper()
        except Exception as exc:
            self.log.error("safety check failed: %s", exc)
            return rules.fail_open

        if verdict.startswith(rules.block_token):
            return False
        if verdict.startswith(rules.allow_token):
            return True

        # Neither word. The provider is offline, or refused, or answered the
        # question instead of judging it. A robot that falls silent mid-lesson
        # is a failed class, and the term list still stood.
        self.log.warning("unusable safety verdict: %s", verdict[:80])
        return rules.fail_open

    def _report(self, text: str, reason: str, session_id: str) -> None:
        self.log.warning("blocked (%s): %s", reason, text)
        self.deps.bus.publish(
            ROBOT_BLOCKED, Blocked(text=text, reason=reason, session_id=session_id)
        )

    def handle(self, _event: str, _payload, _ctx: AgentContext) -> None: ...
