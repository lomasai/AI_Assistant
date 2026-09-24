from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from lomas_core.contracts import (
    QUESTION_ASKED,
    VOLUME_CHANGED,
    SAFETY_CLEARED,
    SAFETY_HALT,
    STORY_REQUESTED,
    QuestionAsked,
    SafetyHalt,
    VolumeChanged,
    StoryRequested,
)
from lomas_core.errors import LomasError



OK = {"ok": True}
TEACHER_SCREEN = "the teacher screen"
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


class Loudness(BaseModel):
    """Either a place on the slider, a nudge along it, or the mute."""

    level: float | None = None
    steps: float = 0.0
    muted: bool | None = None


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

    def teaching() -> bool:
        return system.runner.teaching

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
            "microphone": system.listener.describe() if system.listener else "none",
            "volume": loudness(),
            "signs": system.signs.stats() if system.signs else {},
        }

    @api.post("/session/start")
    def start(body: StartClass) -> dict:
        """Begin a class.

        The rules live in the runner, because the robot can be told to start
        by a voice in the room as well as by this page.
        """
        return {"started": system.runner.start(body.topic, body.language, by=TEACHER_SCREEN)
                or system.cfg.content.default_topic}

    @api.post("/session/stop")
    def stop() -> dict:
        """End the class early. Not a halt: the session closes properly and
        the report is complete."""
        system.runner.stop()
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

    # --- how loud ---------------------------------------------------------

    def dial():
        """The robot's volume knob, if it has a speaker at all. A null voice
        in a test has no player and therefore nothing to turn."""
        return getattr(getattr(system.tts, "player", None), "volume", None)

    def loudness() -> dict:
        knob = dial()
        return {**knob.report(), "available": True} if knob else {"available": False}

    @api.get("/volume")
    def volume() -> dict:
        return loudness()

    @api.post("/volume")
    def set_volume(body: Loudness) -> dict:
        """The slider, the +/- buttons and the mute, in one place.

        The level that comes back is the one the robot settled on, not the
        one asked for: a school can cap the maximum, and some cards only
        have a handful of steps.
        """
        knob = dial()
        if knob is None:
            raise LomasError("this robot has no speaker to turn up")
        if body.muted is not None:
            knob.mute(body.muted)
        if body.level is not None:
            knob.set(body.level)
        elif body.steps:
            knob.nudge(body.steps)

        # On the bus, so a trace shows every hand that moved it, wherever
        # that hand was.
        bus.publish(VOLUME_CHANGED, VolumeChanged(
            level=knob.level, muted=knob.muted, by=TEACHER_SCREEN,
            describe=knob.describe(), at=system.clock.now()))
        return loudness()

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
            # Two different things. A microphone the teacher presses is not
            # a phrase a child can say, and the face must only invite the
            # second when something is actually waiting for it.
            "listening": getattr(system, "listener", None) is not None,
            "wake_listening": False,  # true when a wake loop exists
            "wake_phrase": system.cfg.speech.wake.phrase,
        }

    return api
