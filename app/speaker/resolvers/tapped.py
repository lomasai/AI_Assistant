from __future__ import annotations

from typing import Any

from app.speaker.resolver import RESOLVERS
from app.speaker.types import Heard, Speaker


@RESOLVERS.register("tapped")
class Tapped:
    """The teacher tapped a name. First in the chain, always.

    Whatever the robot believes about faces and voices, the person standing
    in the room can see who spoke, and their answer is the right one.
    """

    name = "tapped"

    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def resolve(self, heard: Heard, _deps: Any) -> Speaker | None:
        student_id, name = heard.tapped
        if not student_id:
            return None
        return Speaker(student_id=student_id, name=name or heard.named(student_id), how=self.name)
