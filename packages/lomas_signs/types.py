from __future__ import annotations

from dataclasses import dataclass

# What a child can show the robot without saying anything: a hand held in a
# shape, or a printed card. Both come back as a box in the frame and a
# meaning, so everything downstream treats them the same way.

TURNS = 4          # a square card has four ways up
DEGREES = 360.0
HALF = 2


@dataclass(frozen=True, slots=True)
class Box:
    x: int
    y: int
    w: int
    h: int

    @property
    def centre(self) -> tuple[float, float]:
        return self.x + self.w / HALF, self.y + self.h / HALF

    def scaled(self, factor: float) -> "Box":
        if factor == 1.0:
            return self
        return Box(x=int(self.x * factor), y=int(self.y * factor),
                   w=int(self.w * factor), h=int(self.h * factor))


@dataclass(frozen=True, slots=True)
class Sign:
    """One hand, held in a shape the reader recognises.

    `name` is whatever the reader calls it - the meaning is not decided here,
    because which sign means "I want to ask" is a school's choice and lives
    in config.
    """

    name: str
    box: Box
    score: float = 0.0
    at: float = 0.0


@dataclass(frozen=True, slots=True)
class Card:
    """A printed marker, and which way up it is being held.

    The id says who: a card belongs to one child. The turn says what: a
    square card held with a different edge up is a different answer, which
    is how a whole class answers a question in one frame.
    """

    marker_id: int
    box: Box
    turn: int = 0
    at: float = 0.0

    @property
    def upright(self) -> bool:
        return self.turn == 0
