from __future__ import annotations

from typing import Any

from app.speaker.resolver import RESOLVERS
from app.speaker.types import Heard, Speaker

ALONE = 1


@RESOLVERS.register("card")
class CardHeldUp:
    """Whoever is holding their card up.

    The only signal in the chain that a child gives on purpose. A face in
    view is a guess about who is speaking; a card held up is a child saying
    "this is me", which is why it sits above everything the robot works out
    for itself and below only the teacher's own tap.

    It is also the answer for a school that will not store faces: a number
    on a piece of paper is not a biometric.
    """

    name = "card"

    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def resolve(self, heard: Heard, _deps: Any) -> Speaker | None:
        # Two cards up is two children with something to say, and picking
        # one of them is how an answer lands against the wrong name.
        if len(heard.cards) != ALONE:
            return None
        return Speaker(student_id=heard.cards[0], how=self.name)
