from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter
from pydantic import BaseModel

from lomas_core.contracts import (
    QUIZ_ANSWERED,
    TEACHER_NUDGING,
    TEACHER_SPEAKER,
    NudgingSet,
    QuizAnswered,
    SpeakerSet,
)
from lomas_core.errors import LomasError

OK = {"ok": True}
SPEAKER = "speaker"
UNMARKED = None


class Consent(BaseModel):
    """Who agreed, and to what. `granted_by` is a person's name, not a
    checkbox: the server refuses without it."""

    granted_by: str = ""
    document_ref: str = ""


class StartEnrolment(BaseModel):
    name: str
    roll_no: str
    consent: Consent = Consent()


class Nudging(BaseModel):
    enabled: bool


class Speaker(BaseModel):
    student_id: str
    student_name: str = ""


class Answer(BaseModel):
    response: str
    question_id: str = ""
    student_id: str = ""
    latency_ms: int = 0


def router(system) -> APIRouter:
    api = APIRouter()
    bus = system.bus

    def scope():
        return system.orchestrator.scope

    def session_id() -> str:
        ctx = system.orchestrator.ctx
        return ctx.session_id if ctx else ""

    def service():
        if system.enrolment is None:
            raise LomasError("enrolment is switched off in config")
        return system.enrolment

    # --- enrolment --------------------------------------------------------

    @api.get("/enrol/students")
    def enrolled() -> dict:
        return {"students": service().enrolled(scope())}

    @api.post("/enrol/start")
    def start(body: StartEnrolment) -> dict:
        return service().start(
            scope(),
            name=body.name,
            roll_no=body.roll_no,
            granted_by=body.consent.granted_by,
            document_ref=body.consent.document_ref,
        )

    @api.post("/enrol/{enrolment_id}/frame")
    def frame(enrolment_id: str) -> dict:
        """Feedback, never a frame. The image is embedded and dropped inside
        the call below; there is no route by which it could come back."""
        return asdict(service().add_frame(enrolment_id))

    @api.post("/enrol/{enrolment_id}/finish")
    def finish(enrolment_id: str) -> dict:
        return service().finish(scope(), enrolment_id)

    @api.post("/enrol/{enrolment_id}/cancel")
    def cancel(enrolment_id: str) -> dict:
        service().cancel(enrolment_id)
        return OK

    @api.delete("/enrol/students/{student_id}")
    def forget(student_id: str) -> dict:
        return service().forget(scope(), student_id)

    # --- live control -----------------------------------------------------

    @api.post("/skip")
    def skip() -> dict:
        return {"skipped": system.extras["machine"].skip()}

    @api.post("/nudging")
    def nudging(body: Nudging) -> dict:
        bus.publish(TEACHER_NUDGING, NudgingSet(enabled=body.enabled, session_id=session_id()))
        return {"enabled": body.enabled}

    @api.post("/speaker")
    def speaker(body: Speaker) -> dict:
        """Tap-to-attribute. Voice alone will not tell you which child spoke
        in a room of forty, so the person in the room decides."""
        ctx = system.orchestrator.ctx
        if ctx is not None:
            ctx.notes[SPEAKER] = (body.student_id, body.student_name or _name(system, body))
        bus.publish(
            TEACHER_SPEAKER,
            SpeakerSet(student_id=body.student_id, student_name=body.student_name,
                       session_id=session_id()),
        )
        return OK

    @api.post("/answer")
    def answer(body: Answer) -> dict:
        """A child's answer, attributed by the teacher. Left unmarked here on
        purpose - the quizmaster reads it, and free text takes a model."""
        ctx = system.orchestrator.ctx
        student_id = body.student_id or (ctx.notes.get(SPEAKER, ("", ""))[0] if ctx else "")
        if not student_id:
            raise LomasError("no student to attribute this answer to; tap a name first")

        bus.publish(
            QUIZ_ANSWERED,
            QuizAnswered(
                session_id=session_id(),
                question_id=body.question_id or _posed(ctx),
                student_id=student_id,
                response=body.response,
                correct=UNMARKED,
                latency_ms=body.latency_ms,
            ),
        )
        return OK

    # --- the report -------------------------------------------------------

    @api.get("/sessions")
    def sessions() -> dict:
        return {"sessions": system.report.recent(scope())}

    @api.get("/report/{session_id_}")
    def report(session_id_: str) -> dict:
        return system.report.build(scope(), session_id_)

    return api


def _name(system, body: Speaker) -> str:
    found = system.repos["student"].get(system.orchestrator.scope, body.student_id)
    return found["name"] if found else ""


def _posed(ctx) -> str:
    return ctx.notes.get("quiz_posed") or "" if ctx else ""
