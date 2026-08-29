from __future__ import annotations

import threading
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from lomas_core.contracts import (
    QUESTION_ASKED,
    SAFETY_CLEARED,
    SAFETY_HALT,
    STORY_REQUESTED,
    QuestionAsked,
    SafetyHalt,
    StoryRequested,
)
from lomas_core.errors import LomasError

from app.flow.states import SessionState

OK = {"ok": True}
TEACHER = "teacher"
IDLE = "idle"


class Ask(BaseModel):
    text: str
    student_id: str = ""
    student_name: str = ""


class Story(BaseModel):
    topic: str = ""


class Halt(BaseModel):
    reason: str = Field(default=TEACHER)


class StartClass(BaseModel):
    topic: str = ""
    language: str = ""
    teacher: str = ""


def router(system) -> APIRouter:
    """Teacher controls, and the state a surface needs on load.

    Everything here either reads or publishes. The one exception is pause and
    resume, which have to reach the machine within a tick because the teacher
    pressing pause is the most important control in the product.
    """
    api = APIRouter()
    bus = system.bus
    running: list[threading.Thread] = []

    def teaching() -> bool:
        return bool(running) and running[0].is_alive()

    @api.get("/state")
    def state() -> dict[str, Any]:
        machine = system.extras["machine"]
        ctx = system.orchestrator.ctx
        return {
            "mode": system.cfg.runtime.mode,
            "teaching": teaching(),
            "state": machine.state.value,
            "step": machine.current,
            "halted_because": machine.halt_reason,
            "session_id": ctx.session_id if ctx else "",
            "language": ctx.language if ctx else system.cfg.content.language,
            "topic": ctx.topic if ctx else "",
            "title": ctx.lesson.title if ctx else "",
            "present": ctx.present if ctx else {},
            # From the repository, not from the session. A teacher picking a
            # child to ask needs the class list before the class starts.
            "roster": [
                {"id": s["id"], "name": s["name"], "roll_no": s["roll_no"]}
                for s in system.repos["student"].list_for_class(system.orchestrator.scope)
            ],
            "agents": system.agents.names() if system.agents else [],
            "vision": system.vision.stats() if system.vision else {},
        }

    @api.post("/session/start")
    def start(body: StartClass) -> dict:
        """Begin a class.

        On its own thread, because a lesson takes forty minutes and the
        surfaces have to stay answerable throughout.
        """
        if teaching():
            raise LomasError("a class is already running")

        machine = system.extras["machine"]
        if machine.state is SessionState.HALTED:
            raise LomasError("the robot is halted; clear it before starting a class")

        thread = threading.Thread(
            target=system.orchestrator.run,
            kwargs={"topic": body.topic, "language": body.language},
            name="class",
            daemon=True,
        )
        running.clear()
        running.append(thread)
        thread.start()
        return {"started": body.topic or system.cfg.content.default_topic}

    @api.post("/session/stop")
    def stop() -> dict:
        """End the class early. Not a halt: the session closes properly and
        the report is complete."""
        machine = system.extras["machine"]
        machine.finish()
        return OK

    @api.get("/topics")
    def topics() -> dict:
        language = system.cfg.content.language
        pack = system.content.load(language)
        return {
            "language": language,
            "topics": [
                {"id": lesson.id, "title": lesson.title, "segments": len(lesson.segments)}
                for lesson in pack.lessons.values()
            ],
        }

    @api.post("/pause")
    def pause() -> dict:
        system.orchestrator.pause()
        return OK

    @api.post("/resume")
    def resume() -> dict:
        system.orchestrator.resume()
        return OK

    @api.post("/halt")
    def halt(body: Halt) -> dict:
        bus.publish(SAFETY_HALT, SafetyHalt(reason=body.reason, at=system.clock.now()))
        return OK

    @api.post("/clear")
    def clear() -> dict:
        bus.publish(SAFETY_CLEARED, {"reason": TEACHER})
        return OK

    @api.post("/ask")
    def ask(body: Ask) -> dict:
        """The teacher relaying a question, or a dashboard standing in for
        speech. Attribution comes with it rather than being guessed."""
        ctx = system.orchestrator.ctx
        bus.publish(
            QUESTION_ASKED,
            QuestionAsked(
                session_id=ctx.session_id if ctx else "",
                text=body.text,
                student_id=body.student_id,
                student_name=body.student_name,
            ),
        )
        return OK

    @api.post("/story")
    def story(body: Story) -> dict:
        ctx = system.orchestrator.ctx
        bus.publish(
            STORY_REQUESTED,
            StoryRequested(
                session_id=ctx.session_id if ctx else "",
                topic=body.topic,
                language=ctx.language if ctx else system.cfg.content.language,
            ),
        )
        return OK

    @api.get("/display")
    def display() -> dict:
        """What a surface needs to lay itself out. Fetched rather than
        written into the CSS, so the panel size and the engaged/drifting
        boundary stay config like everything else."""
        return {
            **system.cfg.display.model_dump(),
            "attention_threshold": system.cfg.attention.threshold,
            # Whether anything is actually driving the wake engine. The face
            # must not invite a child to say a phrase nobody is listening for:
            # a screen that asks for something and then ignores it is worse
            # than a screen that asks for nothing.
            "listening": getattr(system, "listener", None) is not None,
            "wake_phrase": system.cfg.speech.wake.phrase,
        }

    return api
