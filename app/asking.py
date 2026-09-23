from __future__ import annotations

import threading

from lomas_core import logging as log
from lomas_core.clock import Clock
from lomas_core.contracts import (
    HAND_UP,
    ROBOT_SAY,
    SESSION_CLOSED,
    STEP_ENTERED,
    HandUp,
    Utterance,
)
from lomas_core.errors import LomasError
from lomas_core.events import EventBus
from lomas_core.schema import Config

YES_CHILD = "yes_child"
WAIT_YOUR_TURN = "wait_your_turn"
ASKING = "asking"
NOBODY = ""
NONE_YET = 0


class Asking:
    """A child putting a hand up in the middle of the lesson.

    The rule that makes this sound like a teacher rather than a machine: the
    robot finishes the sentence it is saying, then stops. What it had not
    said is kept, and once the question has been answered the lesson picks
    up exactly there.

    The rest is classroom management, and all of it is config. A queue, so
    the first hand up gets the turn. A cap per step, so one question does
    not eat a lesson. A cooldown, so the same confident child does not get
    every turn and the quiet ones get none.
    """

    def __init__(self, cfg: Config, bus: EventBus, clock: Clock, voice, listener,
                 prompts, teaching=None) -> None:
        self.cfg = cfg
        self.bus = bus
        self.clock = clock
        self.voice = voice
        self.listener = listener
        self.prompts = prompts
        # Whether a class is running at all. A hand up in an empty room is
        # somebody waving at the robot.
        self.teaching = teaching or (lambda: False)
        self.log = log.get("asking")

        self.taken = 0
        self.refused = 0
        self._this_step = NONE_YET
        self._last_turn: dict[str, float] = {}
        self._busy = threading.Lock()

        bus.subscribe(HAND_UP, self._on_hand)
        bus.subscribe(STEP_ENTERED, self._on_step)
        bus.subscribe(SESSION_CLOSED, self._on_closed)

    @property
    def enabled(self) -> bool:
        return bool(self.cfg.signs.asking.enabled and self.voice is not None
                    and self.listener is not None and self.listener.available)

    # --- whose turn --------------------------------------------------------

    def _on_step(self, _event: str, _step) -> None:
        # Per step, not per class: two questions between one idea and the
        # next is a conversation, twenty is a different lesson.
        self._this_step = NONE_YET

    def _on_closed(self, _event: str, _closed) -> None:
        self._this_step = NONE_YET
        self._last_turn.clear()

    def _on_hand(self, _event: str, up: HandUp) -> None:
        if not self.enabled or not self.teaching() or not self._may(up):
            return
        threading.Thread(target=self._take, args=(up,), name="asking", daemon=True).start()

    def _may(self, up: HandUp) -> bool:
        if self._busy.locked():
            # Somebody already has the floor. Theirs is the turn; this hand
            # stays up and will be seen again in a second.
            return False
        if self._this_step >= self.cfg.signs.asking.per_step:
            self.refused += 1
            self.log.debug("not now: %s questions already in this step", self._this_step)
            return False

        since = self._last_turn.get(up.student_id or NOBODY)
        if since is not None and self.clock.now() - since < self.cfg.signs.asking.cooldown_seconds:
            self.refused += 1
            self.log.debug("not again so soon: %s", up.student_name or up.student_id)
            return False
        return True

    # --- the turn itself ---------------------------------------------------

    def _take(self, up: HandUp) -> None:
        if not self._busy.acquire(blocking=False):
            return
        try:
            self._this_step += 1
            self._last_turn[up.student_id or NOBODY] = self.clock.now()
            self.taken += 1
            self.log.info("hand up from %s (%s); finishing the sentence",
                          up.student_name or "somebody", up.by)

            # Finish the sentence, then stop. Everything after this happens
            # in a room where the robot has gone quiet on purpose.
            if self.cfg.signs.asking.finish_the_sentence:
                self.voice.yield_now()

            self._say(YES_CHILD, name=up.student_name or NOBODY)
            self._hear(up)
        except LomasError as exc:
            self.log.error("the turn failed: %s", exc)
        finally:
            # Always: a lesson that stopped for a child and never started
            # again is worse than one that never stopped.
            self.voice.resume()
            self._busy.release()

    def _hear(self, up: HandUp) -> None:
        """Listen, and publish it as a question like any other.

        Attribution comes from whoever raised the hand, not from guessing:
        the card said who, or the face beside the hand did.
        """
        self.listener.listen(
            student_id=up.student_id, student_name=up.student_name,
            seconds=self.cfg.signs.asking.listen_seconds, as_question=True,
            attribute=not up.student_id,
        )

    def _say(self, prompt: str, **values) -> None:
        if self.prompts is None:
            return
        try:
            text = self.prompts.line(prompt, self.cfg.content.language, **values)
        except Exception as exc:  # a missing prompt file must not eat a turn
            self.log.debug("no %s prompt: %s", prompt, exc)
            return
        self.bus.publish(ROBOT_SAY, Utterance(text=text, language=self.cfg.content.language,
                                              reason=ASKING))
