from __future__ import annotations

import threading

from lomas_core import logging as log
from lomas_core.clock import Clock
from lomas_core.contracts import STEP_ENTERED, STEP_EXITED, StepChanged
from lomas_core.events import EventBus
from lomas_core.schema import FlowConfig

from app.flow.states import SessionState, StepResult
from app.flow.step import FlowStep


class Machine:
    """Walks the configured steps in order.

    Pause, resume and halt are interrupts that apply from any step. Pause has
    to take effect within a tick, because the teacher pressing it is the most
    important control in the product.
    """

    def __init__(self, steps: list[FlowStep], cfg: FlowConfig, bus: EventBus, clock: Clock) -> None:
        self.steps = steps
        self.cfg = cfg
        self.bus = bus
        self.clock = clock
        self.state = SessionState.IDLE
        self.current: str = ""
        self.halt_reason = ""
        self._resume = threading.Event()
        self._resume.set()
        self.log = log.get("flow")

    def pause(self) -> None:
        if self.state is SessionState.RUNNING:
            self.state = SessionState.PAUSED
            self._resume.clear()

    def resume(self) -> None:
        if self.state is SessionState.PAUSED:
            self.state = SessionState.RUNNING
            self._resume.set()

    def halt(self, reason: str) -> None:
        """Latches, like the physical e-stop it usually comes from. Nothing
        runs again until someone clears it."""
        self.state = SessionState.HALTED
        self.halt_reason = reason
        self._resume.set()

    def clear(self) -> None:
        if self.state is SessionState.HALTED:
            self.state = SessionState.IDLE
            self.halt_reason = ""

    def run(self, ctx) -> SessionState:
        if self.state is SessionState.HALTED:
            # A halt raised before the class began must not be discarded by
            # starting one. The e-stop is latched until it is cleared.
            self.log.warning("refusing to start: halted (%s)", self.halt_reason)
            return self.state

        self.state = SessionState.RUNNING

        for step in self.steps:
            if self.state is SessionState.HALTED:
                break

            self.current = step.name
            self.log.info("-> %s", step.name)

            # enter() is where a step puts its subscriptions up, so the event
            # is published after it. STEP_ENTERED means "this step is running
            # and listening", which is what any subscriber actually needs.
            step.enter(ctx)
            self.bus.publish(STEP_ENTERED, StepChanged(ctx.session_id, step.name, self.clock.now()))

            self._drive(step, ctx)

            step.exit(ctx)
            self.bus.publish(STEP_EXITED, StepChanged(ctx.session_id, step.name, self.clock.now()))

        self.current = ""
        if self.state is not SessionState.HALTED:
            self.state = SessionState.CLOSED
        return self.state

    def _drive(self, step: FlowStep, ctx) -> None:
        started = self.clock.now()
        budget = self.cfg.stage_timeout_seconds.get(step.name, self.cfg.default_timeout_seconds)

        while True:
            if self.state is SessionState.HALTED:
                return

            if self.state is SessionState.PAUSED:
                self._resume.wait(self.cfg.pause_poll_seconds)
                continue

            result = step.tick(ctx, self.clock.now())
            if result is StepResult.DONE:
                return
            if result is StepResult.ABORT:
                self.halt(f"{step.name} aborted")
                return

            if self.clock.now() - started >= budget:
                self.log.debug("%s hit its %ss budget", step.name, budget)
                return

            self.clock.sleep(self.cfg.tick_seconds)
