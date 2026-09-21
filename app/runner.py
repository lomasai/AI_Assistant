from __future__ import annotations

import threading

from lomas_core import logging as log
from lomas_core.errors import LomasError

from app.author import clean_topic
from app.flow.states import SessionState

CLASS = "class"


class ClassRunner:
    """Starts and ends a class, from wherever the instruction came.

    The teacher's screen was the only thing that could do this, which made a
    laptop part of the robot. A spoken "start the class" has exactly the same
    right to it, so both go through here and the rules - one class at a time,
    not while halted - are written once.
    """

    def __init__(self, system) -> None:
        self.system = system
        self.log = log.get("class")
        self._thread: threading.Thread | None = None

    @property
    def teaching(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, topic: str = "", language: str = "", by: str = "") -> str:
        if self.teaching:
            raise LomasError("a class is already running")

        machine = self.system.extras["machine"]
        if machine.state is SessionState.HALTED:
            raise LomasError("the robot is halted; clear it before starting a class")

        # What a child said, reduced to its subject. Empty is not a mistake:
        # it means the robot asks the class what to learn.
        wanted = clean_topic(topic, self.system.cfg.content.author) if topic else ""

        self._thread = threading.Thread(
            target=self.system.orchestrator.run,
            kwargs={"topic": wanted, "language": language},
            name=CLASS,
            daemon=True,
        )
        self._thread.start()
        self.log.info("class started%s%s", f" on {wanted}" if wanted else "",
                      f", asked for by {by}" if by else "")
        return wanted

    def stop(self, wait: bool = True) -> str:
        """End the class properly, so the report is complete. Not a halt.

        Waits for the lesson to actually stop by default: the robot is
        usually being switched off next, and a class still writing its report
        into a database that has just been closed is the fault that taught us
        to do this.
        """
        topic = self.system.orchestrator.ctx.topic if self.system.orchestrator.ctx else ""
        self.system.extras["machine"].finish()
        if wait and self._thread is not None:
            self._thread.join(timeout=self.system.cfg.flow.stop_wait_seconds)
        return topic
