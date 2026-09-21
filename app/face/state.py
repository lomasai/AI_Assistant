from __future__ import annotations

import threading
from dataclasses import dataclass, field

from lomas_core.contracts import (
    ATTENDANCE_MARKED,
    LESSON_SEGMENT,
    QUESTION_ASKED,
    QUIZ_POSED,
    ROBOT_SAY,
    ROBOT_SPOKE,
    ROBOT_STATE,
    SESSION_CLOSED,
    SESSION_OPENED,
    STUDENT_DISENGAGED,
    STUDENT_IDENTIFIED,
    STUDENT_LEFT,
)
from lomas_core.events import EventBus

SLEEPING = "sleeping"
LISTENING = "listening"
SPEAKING = "speaking"
THINKING = "thinking"
ASKING = "asking"
IDLE = "idle"  # what the microphone says when it closes

AWAY = "away"
ENGAGED = "engaged"
DRIFTING = "drifting"
TALKING_TO = "speaking"

FIRST_NAME = 0


@dataclass
class Child:
    name: str
    mood: str = AWAY


@dataclass
class Look:
    """Everything a face needs to draw itself, and nothing else.

    Separate from the window on purpose: what the robot should look like is
    decided by events and can be tested without a screen, while the drawing
    is a loop that cannot. It is also what lets a second kind of face - a
    browser, a panel of LEDs - read the same thing.
    """

    state: str = SLEEPING
    line: str = ""
    addressing: str = ""
    options: tuple[str, ...] = ()
    children: dict[str, Child] = field(default_factory=dict)
    version: int = 0  # bumped on every change, so a window redraws only then


class FaceState:
    """What the robot's face is doing, kept up to date from the event bus.

    It subscribes and nothing else. There is no route from here to a step, an
    agent or the orchestrator, which is what makes the face safe to remove.
    """

    def __init__(self, bus: EventBus) -> None:
        self.look = Look()
        self._lock = threading.RLock()

        handlers = {
            SESSION_OPENED: self._opened,
            SESSION_CLOSED: self._closed,
            ATTENDANCE_MARKED: self._present,
            STUDENT_IDENTIFIED: self._identified,
            STUDENT_LEFT: self._left,
            STUDENT_DISENGAGED: self._drifting,
            ROBOT_SAY: self._saying,
            ROBOT_SPOKE: self._spoke,
            ROBOT_STATE: self._robot_state,
            QUESTION_ASKED: self._asked,
            LESSON_SEGMENT: self._segment,
            QUIZ_POSED: self._quiz,
        }
        for event, handler in handlers.items():
            bus.subscribe(event, handler)

    def snapshot(self) -> Look:
        with self._lock:
            return Look(
                state=self.look.state,
                line=self.look.line,
                addressing=self.look.addressing,
                options=self.look.options,
                children={who: Child(child.name, child.mood)
                          for who, child in self.look.children.items()},
                version=self.look.version,
            )

    # --- the class --------------------------------------------------------

    def _opened(self, _event, _payload) -> None:
        self._set(state=LISTENING, line="", options=())

    def _closed(self, _event, _payload) -> None:
        with self._lock:
            self.look.children.clear()
        self._set(state=SLEEPING, line="", addressing="", options=())

    def _present(self, _event, marked) -> None:
        self._child(getattr(marked, "student_id", ""), getattr(marked, "name", ""), AWAY)

    def _identified(self, _event, who) -> None:
        self._child(who.student_id, "", ENGAGED)

    def _left(self, _event, who) -> None:
        self._child(who.student_id, "", AWAY)

    def _drifting(self, _event, who) -> None:
        self._child(getattr(who, "student_id", ""), "", DRIFTING)

    # --- the robot --------------------------------------------------------

    def _saying(self, _event, utterance) -> None:
        self._set(state=SPEAKING, line=utterance.text, addressing=utterance.student_name,
                  options=())
        if utterance.student_name:
            self._mood_of(utterance.student_name, TALKING_TO)

    def _spoke(self, _event, _utterance) -> None:
        with self._lock:
            resting = LISTENING if self.look.children or self.look.line else SLEEPING
        self._set(state=resting)

    def _robot_state(self, _event, payload) -> None:
        # Both the microphone opening and its closing leave the face
        # listening: a class that has started is a robot paying attention.
        heard = payload.get("state") if isinstance(payload, dict) else getattr(payload, "state", "")
        if heard in (LISTENING, IDLE):
            self._set(state=LISTENING)

    def _asked(self, _event, asked) -> None:
        self._set(state=THINKING, line=asked.text, addressing=asked.student_name, options=())

    def _segment(self, _event, segment) -> None:
        shown = getattr(segment, "display", "") or getattr(segment, "say", "")
        self._set(state=SPEAKING, line=shown, options=())

    def _quiz(self, _event, posed) -> None:
        self._set(state=ASKING, line=posed.text, addressing="",
                  options=tuple(getattr(posed, "options", ()) or ()))

    # --- the two ways anything changes ------------------------------------

    def _set(self, **fields) -> None:
        with self._lock:
            for name, value in fields.items():
                setattr(self.look, name, value)
            self.look.version += 1

    def _child(self, student_id: str, name: str, mood: str) -> None:
        if not student_id:
            return
        with self._lock:
            child = self.look.children.get(student_id)
            if child is None:
                self.look.children[student_id] = Child(name=first_name(name), mood=mood)
            else:
                child.mood = mood
                child.name = child.name or first_name(name)
            self.look.version += 1

    def _mood_of(self, name: str, mood: str) -> None:
        wanted = first_name(name)
        with self._lock:
            for child in self.look.children.values():
                if child.name == wanted:
                    child.mood = mood
                elif child.mood == TALKING_TO:
                    child.mood = ENGAGED
            self.look.version += 1


def first_name(name: str) -> str:
    """A face has room for one word, and a class has two Sharmas."""
    return name.split()[FIRST_NAME] if name else ""
