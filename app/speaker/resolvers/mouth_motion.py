from __future__ import annotations

from typing import Any

from app.speaker.resolver import RESOLVERS
from app.speaker.types import Heard, Speaker


@RESOLVERS.register("mouth_motion")
class MouthMotion:
    """The face whose mouth was moving while the sound came in.

    Last of the ways that do not ask anybody anything, and the least certain:
    two children sitting close, both fidgeting, and it will sometimes pick
    the wrong one. So it only answers when one face is clearly moving more
    than the others - `mouth_min_score` above the rest - and otherwise says
    nothing and lets the robot ask.
    """

    name = "mouth_motion"

    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def resolve(self, heard: Heard, _deps: Any) -> Speaker | None:
        moving = heard.mouths
        if len(moving) < 2:
            # One face is `single_face`'s job, and nobody's is nobody's.
            return None

        ranked = sorted(moving.items(), key=lambda pair: pair[1], reverse=True)
        (student_id, loudest), (_other, next_one) = ranked[0], ranked[1]
        if loudest < self.cfg.mouth_min_score or loudest - next_one < self.cfg.mouth_clear_by:
            return None
        return Speaker(student_id=student_id, how=self.name)
