from __future__ import annotations

from typing import Any

from app.speaker.resolver import RESOLVERS
from app.speaker.types import Heard, Speaker


@RESOLVERS.register("recent")
class Recent:
    """Whoever spoke a moment ago is probably still speaking.

    What makes saying your name bearable: you say it once and then hold a
    conversation. A child who is still visible keeps the turn until the
    window runs out or somebody else is identified.
    """

    name = "recent"

    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def resolve(self, heard: Heard, _deps: Any) -> Speaker | None:
        last = heard.last
        if last is None or heard.since_last > self.cfg.recent_seconds:
            return None
        # Someone else, alone in front of the camera, is not the last speaker.
        if heard.visible and last.student_id not in heard.visible:
            return None
        return Speaker(student_id=last.student_id, name=last.name, how=self.name)
