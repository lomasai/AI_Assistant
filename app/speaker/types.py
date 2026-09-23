from __future__ import annotations

from dataclasses import dataclass, field

NOBODY = ""


@dataclass(frozen=True, slots=True)
class Speaker:
    """Who said it, and how that was decided.

    `how` is kept because the teacher's screen has to be able to show it. A
    robot that says "Akshay" with no way to see why it thinks so is one a
    teacher cannot correct.
    """

    student_id: str = NOBODY
    name: str = NOBODY
    how: str = NOBODY
    text: str = NOBODY  # the words, with any "I am Akshay" taken off

    def __bool__(self) -> bool:
        return bool(self.student_id)


@dataclass(frozen=True, slots=True)
class Heard:
    """One turn, and everything a resolver may look at."""

    text: str
    session_id: str = NOBODY
    language: str = NOBODY
    tapped: tuple[str, str] = (NOBODY, NOBODY)
    roster: list[dict] = field(default_factory=list)
    visible: list[str] = field(default_factory=list)  # student ids in front of the camera
    # Student ids holding a printed card up right now. A card is the one
    # signal in here that a child chooses to give, which is why it outranks
    # everything the robot works out for itself.
    cards: list[str] = field(default_factory=list)
    # How much each visible child's mouth moved just now, by student id.
    mouths: dict[str, float] = field(default_factory=dict)
    last: Speaker | None = None
    since_last: float = 0.0

    def named(self, student_id: str) -> str:
        for row in self.roster:
            if row["id"] == student_id:
                return row["name"]
        return NOBODY
