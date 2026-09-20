from __future__ import annotations

from typing import Any

from lomas_core.contracts import ROBOT_SAY, Utterance

from app.speaker.resolver import RESOLVERS
from app.speaker.types import Heard

ASK_PROMPT = "who_is_asking"


@RESOLVERS.register("ask")
class Ask:
    """Last in the chain: the robot says it does not know who spoke.

    It never returns a speaker - asking is not knowing. The answer arrives on
    the next turn, where `spoken_name` picks the name out of it. Put this
    last or it will interrupt a child the robot could have recognised.
    """

    name = "ask"

    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def resolve(self, heard: Heard, deps: Any) -> None:
        try:
            line = deps.prompts.line(ASK_PROMPT, heard.language)
        except Exception as exc:  # a missing prompt file must not eat the turn
            deps.log.debug("no %s prompt: %s", ASK_PROMPT, exc)
            return None

        deps.bus.publish(
            ROBOT_SAY,
            Utterance(text=line, language=heard.language, session_id=heard.session_id,
                      reason=self.name, blocking=False),
        )
        return None
