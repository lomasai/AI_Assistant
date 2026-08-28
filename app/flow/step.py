from __future__ import annotations

from typing import Protocol, runtime_checkable

from lomas_core.registry import Registry

from app.flow.states import StepResult


@runtime_checkable
class FlowStep(Protocol):
    name: str

    def enter(self, ctx) -> None: ...

    def tick(self, ctx, now: float) -> StepResult: ...

    def exit(self, ctx) -> None: ...


STEPS: Registry[FlowStep] = Registry("flow step")


class BaseStep:
    """Default no-op lifecycle so a step only writes the part it cares about.

    A step may publish events and read its context. It may not call another
    step, reach into the orchestrator, or invoke an agent directly - that is
    what keeps any one of them removable from flow.sequence.
    """

    name = ""

    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def enter(self, ctx) -> None: ...

    def tick(self, ctx, now: float) -> StepResult:
        return StepResult.DONE

    def exit(self, ctx) -> None: ...
