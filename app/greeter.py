from __future__ import annotations

import threading

from lomas_core import logging as log
from lomas_core.clock import Clock
from lomas_core.contracts import STRANGER_SEEN, StrangerSeen
from lomas_core.errors import LomasError
from lomas_core.events import EventBus
from lomas_core.schema import Config

ASK_NAME = "ask_name"
COACH = "enrol_coach"
WELCOME = "enrol_welcome"
GAVE_UP = "enrol_gave_up"
ROBOT = "robot"


class Greeter:
    """Introduces the robot to a child it does not know, and enrols them.

    Enrolment was a form on the teacher's screen, which made a laptop part of
    the robot. The camera already notices a face it cannot name; the voice
    can already ask a question and hear the answer; the sweep already coaches
    a head left and right. This is those three things in a row.

    Off unless `enrolment.by_voice` says otherwise: who may consent to a
    child's face being stored is a school's decision, not a default.
    """

    def __init__(self, cfg: Config, bus: EventBus, clock: Clock, enrolment, listener,
                 prompts, say, scope_of, busy=None) -> None:
        self.cfg = cfg
        self.bus = bus
        self.clock = clock
        self.enrolment = enrolment
        self.listener = listener
        self.prompts = prompts
        self.say = say
        self.scope_of = scope_of
        # Whether something else has the robot's attention. A class in
        # progress is not the moment to ask a stranger their name.
        self.busy = busy or (lambda: False)
        self.log = log.get("greeter")

        self.met = 0
        self._working = threading.Lock()
        bus.subscribe(STRANGER_SEEN, self._on_stranger)

    @property
    def enabled(self) -> bool:
        return bool(self.cfg.enrolment.by_voice and self.enrolment is not None
                    and self.listener is not None and self.listener.available)

    def _on_stranger(self, _event: str, seen: StrangerSeen) -> None:
        if not self.enabled or self.busy() or self._working.locked():
            return
        threading.Thread(target=self._meet, args=(seen,), name="greeter", daemon=True).start()

    def _meet(self, seen: StrangerSeen) -> None:
        if not self._working.acquire(blocking=False):
            return
        try:
            name = self._ask_name()
            if not name:
                return
            self._enrol(name)
        except LomasError as exc:
            # A robot that cannot enrol somebody is a robot that teaches them
            # anyway, under no name.
            self.log.error("could not enrol: %s", exc)
            self._line(GAVE_UP, name="")
        finally:
            self._working.release()

    # --- who are you ------------------------------------------------------

    def _ask_name(self) -> str:
        self._line(ASK_NAME, name="")
        heard = self.listener.listen(as_question=False, attribute=False,
                                     seconds=self.cfg.enrolment.name_wait_seconds)
        name = clean_name(heard.get("text", ""), self.cfg.enrolment.name_lead_ins)
        if not name:
            self.log.info("nobody answered; not enrolling")
            return ""
        self.log.info("meeting %s", name)
        return name

    # --- the sweep, coached out loud --------------------------------------

    def _enrol(self, name: str) -> None:
        scope = self.scope_of()
        started = self.enrolment.start(scope, name=name,
                                       roll_no=self.enrolment.next_roll(scope),
                                       granted_by=self.cfg.enrolment.voice_consent_by)
        enrolment_id = started["enrolment_id"]

        try:
            for angle in self.cfg.enrolment.required_angles:
                self._line(COACH, name=name, angle=angle)
                self._sweep(enrolment_id)

            done = self.enrolment.finish(scope, enrolment_id)
        except LomasError:
            self.enrolment.cancel(enrolment_id)
            raise

        self.met += 1
        self.log.info("enrolled %s with %s vectors", name, done.get("vectors", 0))
        self._line(WELCOME, name=name)

    def _sweep(self, enrolment_id: str) -> None:
        """Frames while the child holds one angle. A frame the sweep refuses
        is a blurred one or a repeat, and neither is worth saying anything
        about - the coaching is the same either way."""
        until = self.clock.now() + self.cfg.enrolment.sweep_seconds
        while self.clock.now() < until:
            try:
                self.enrolment.add_frame(enrolment_id)
            except LomasError as exc:
                self.log.debug("frame skipped: %s", exc)
            self.clock.sleep(self.cfg.enrolment.frame_gap_seconds)

    def _line(self, prompt: str, **values) -> None:
        try:
            self.say(self.prompts.line(prompt, self.cfg.content.language, **values))
        except Exception as exc:  # a missing prompt file must not end a meeting
            self.log.debug("no %s prompt: %s", prompt, exc)


def clean_name(heard: str, lead_ins: list[str]) -> str:
    """A name out of whatever was said back.

    "My name is Akshay" and "I am Akshay" and "Akshay" are one answer, and a
    child who says a sentence should not be enrolled as that sentence.
    """
    said = " ".join(heard.split()).strip(" .,!?")
    lowered = said.lower()
    for lead in sorted(lead_ins, key=len, reverse=True):
        if lowered.startswith(lead + " "):
            said = said[len(lead) + 1:].strip(" .,!?")
            break
    words = said.split()
    return " ".join(word.capitalize() for word in words[:2]) if words else ""
