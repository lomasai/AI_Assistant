from __future__ import annotations

from typing import Any

from app.speaker.resolver import RESOLVERS
from app.speaker.types import Heard, Speaker

ALONE = 1


@RESOLVERS.register("single_face")
class SingleFace:
    """One known face in front of the camera, so that is who spoke.

    This is the whole of a one-child demo and a good part of a real lesson,
    and it costs nothing: the camera is already recognising faces for
    attendance.
    """

    name = "single_face"

    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def resolve(self, heard: Heard, _deps: Any) -> Speaker | None:
        if len(heard.visible) != ALONE:
            return None
        return Speaker(student_id=heard.visible[0], how=self.name)
