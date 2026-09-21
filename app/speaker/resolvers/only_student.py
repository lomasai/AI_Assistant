from __future__ import annotations

from typing import Any

from app.speaker.resolver import RESOLVERS
from app.speaker.types import Heard, Speaker

ALONE = 1


@RESOLVERS.register("only_student")
class OnlyStudent:
    """One child in the class, so that is who spoke.

    Not the same as `single_face`, which needs the camera to have recognised
    somebody this second. A child at the side of the robot, or one testing it
    from behind a laptop, is still the only person it could be - and asking
    "who was that?" of a class of one is a robot that is not paying
    attention.
    """

    name = "only_student"

    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def resolve(self, heard: Heard, _deps: Any) -> Speaker | None:
        if len(heard.roster) != ALONE:
            return None
        return Speaker(student_id=heard.roster[0]["id"], name=heard.roster[0]["name"],
                       how=self.name)
