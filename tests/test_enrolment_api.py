"""Enrolment, teacher control and the report.

Two rules the server has to hold on its own, because a browser is a claim and
not a record: nothing is stored without a consent row, and no endpoint ever
returns an image. Both are asserted against the API, not against the UI.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from lomas_core.clock import FakeClock
from lomas_core.config import load
from lomas_core.contracts import (
    LESSON_SEGMENT,
    QUESTION_ASKED,
    QUIZ_ANSWERED,
    ROBOT_SAY,
    STUDENT_DISENGAGED,
    STUDENT_ENROLLED,
    QuestionAsked,
    QuizAnswered,
    StudentDisengaged,
    Utterance,
)
from lomas_core.events import EventBus
from lomas_face.types import Detection, Landmarks
from lomas_speech.types import SpeechHandle
from lomas_vision import Frame

from app import container, seed
from app.flow.states import SessionState
from app.web.server import create_app

TEACHER_JS = (Path("app/web/ui/teacher/teacher.js")).read_text(encoding="utf-8")

WIDTH = 1280
HEIGHT = 720
FACE_PX = 200
SMALL_WIDTH = 640
FACTOR = WIDTH // SMALL_WIDTH

PARENT = "R. Sharma"
ROLL = "11"

HEADLESS = [
    "storage.backend=memory",
    "vision.pipeline.enabled=false",
    "face.detector=mock",
    "face.embedder=mock",
    f"face.downscale_width={SMALL_WIDTH}",
    "speech.tts.engine=null",
    "speech.stt.engine=keyboard",
    "speech.wake.engine=keyboard",
    "llm.provider=offline",
    "flow.attendance_wait_seconds=1",
    "flow.answer_wait_seconds=1",
    "flow.tick_seconds=0.1",
]


def build(*extra: str):
    cfg = load("config", "debug", [*HEADLESS, *extra], use_env=False)
    system = container.build(cfg, clock=FakeClock(), bus=container.event_bus(cfg))
    seed.demo_class(system)
    return system


@pytest.fixture
def system():
    built = build()
    yield built
    built.close()


@pytest.fixture
def client(system):
    with TestClient(create_app(system)) as opened:
        yield opened


# --- a camera the enrolment service can actually read ---------------------


class OneFace:
    """A single face at a known place, painted so the mock embedder produces
    a stable vector, with landmarks that walk left to right across the sweep."""

    def __init__(self) -> None:
        self.turn = 0

    def frame(self) -> Frame:
        image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        image[200 : 200 + FACE_PX, 400 : 400 + FACE_PX] = 140
        return Frame(source_id="head", zone="front", seq=self.turn + 1, ts=0.0, image=image)

    def detection(self) -> Detection:
        x, y = 400 // FACTOR, 200 // FACTOR
        span = FACE_PX // FACTOR
        # Sweeps left, centre, right across successive frames so all three
        # angle buckets fill, the way a child turning their head would.
        nose = x + span // 2 + [-18, 0, 18][self.turn % 3]
        self.turn += 1
        return Detection(
            x=x, y=y, w=span, h=span, confidence=0.99,
            landmarks=Landmarks(
                right_eye=(x + span * 0.3, y + span * 0.35),
                left_eye=(x + span * 0.7, y + span * 0.35),
                nose=(nose, y + span * 0.55),
                right_mouth=(x + span * 0.35, y + span * 0.75),
                left_mouth=(x + span * 0.65, y + span * 0.75),
            ),
        )


class FakeCamera:
    """Stands in for the FrameBus. The service asks for the newest frame and
    nothing else, which is the whole of its camera dependency."""

    def __init__(self, scene: OneFace) -> None:
        self.scene = scene

    def latest(self, _source_id: str):
        return self.scene.frame()


class SweepDetector:
    def __init__(self, scene: OneFace) -> None:
        self.scene = scene

    def detect(self, _image) -> list[Detection]:
        return [self.scene.detection()]


class NoFace:
    def detect(self, _image) -> list[Detection]:
        return []


def with_camera(system) -> OneFace:
    """Give the enrolment service something to look at. Sharpness is measured
    on the real crop, so the quality floor is dropped rather than faked."""
    scene = OneFace()
    system.enrolment.frames = FakeCamera(scene)
    system.enrolment.detector = SweepDetector(scene)
    system.cfg.enrolment.min_quality = 0.0
    system.cfg.enrolment.min_face_px = 1
    return scene


def enrol(client, system, name="Nikhil Rao", roll=ROLL, granted_by=PARENT, frames=9):
    with_camera(system)
    started = client.post("/api/enrol/start", json={
        "name": name, "roll_no": roll, "consent": {"granted_by": granted_by},
    })
    assert started.status_code == 200, started.text
    enrolment_id = started.json()["enrolment_id"]

    feedback = [client.post(f"/api/enrol/{enrolment_id}/frame").json() for _ in range(frames)]
    done = client.post(f"/api/enrol/{enrolment_id}/finish")
    return started.json(), feedback, done


# --- consent, enforced on the server --------------------------------------


def test_enrolment_without_consent_is_refused(client) -> None:
    """The acceptance criterion. A checkbox in a browser is a claim; the
    server will not store a vector without a person's name against it."""
    refused = client.post("/api/enrol/start", json={
        "name": "Nikhil Rao", "roll_no": ROLL, "consent": {"granted_by": ""},
    })

    assert refused.status_code == 400
    assert "consent" in refused.json()["error"]


def test_a_request_with_no_consent_block_at_all_is_refused(client) -> None:
    refused = client.post("/api/enrol/start", json={"name": "Nikhil Rao", "roll_no": ROLL})
    assert refused.status_code == 400


def test_starting_an_enrolment_writes_a_consent_row(client, system) -> None:
    started = client.post("/api/enrol/start", json={
        "name": "Nikhil Rao", "roll_no": ROLL,
        "consent": {"granted_by": PARENT, "document_ref": "form-12"},
    }).json()

    rows = system.repos["consent"].for_student(system.orchestrator.scope, started["student_id"])
    assert len(rows) == 1
    assert rows[0]["granted_by"] == PARENT
    assert rows[0]["document_ref"] == "form-12"


def test_consent_withdrawn_mid_sweep_stores_nothing(client, system) -> None:
    """Checked again at the end on purpose. A revoke during the sweep has to
    mean the vectors are never written, not written and tidied up later."""
    with_camera(system)
    started = client.post("/api/enrol/start", json={
        "name": "Nikhil Rao", "roll_no": ROLL, "consent": {"granted_by": PARENT},
    }).json()

    for _ in range(6):
        client.post(f"/api/enrol/{started['enrolment_id']}/frame")

    scope = system.orchestrator.scope
    system.repos["consent"].revoke(scope, started["student_id"], "face_recognition")

    refused = client.post(f"/api/enrol/{started['enrolment_id']}/finish")
    assert refused.status_code == 400
    assert "withdrawn" in refused.json()["error"]
    assert system.repos["embedding"].for_student(scope, started["student_id"]) == []


# --- no image, ever -------------------------------------------------------


IMAGE_WORDS = ("image", "photo", "picture", "jpeg", "jpg", "png", "frame_data", "thumbnail")


def test_no_enrolment_endpoint_returns_image_bytes(client, system) -> None:
    """The acceptance criterion, checked over the wire rather than trusted.
    Frames are embedded and dropped inside add_frame; nothing downstream has
    a field one could travel in."""
    started, feedback, done = enrol(client, system)

    for response in [started, *feedback, done.json()]:
        body = json.dumps(response).lower()
        assert not [word for word in IMAGE_WORDS if word in body], body
        assert "base64" not in body
        assert "data:" not in body


def test_the_enrolment_result_is_vectors_and_a_count(client, system) -> None:
    _started, _feedback, done = enrol(client, system)
    body = done.json()

    assert body["vectors"] > 1
    assert set(body["angles"]) <= {"left", "centre", "right"}
    assert set(body) == {"student_id", "name", "vectors", "angles", "quality"}


def test_the_vectors_reach_the_database(client, system) -> None:
    _started, _feedback, done = enrol(client, system)
    scope = system.orchestrator.scope
    rows = system.repos["embedding"].for_student(scope, done.json()["student_id"])

    assert len(rows) == done.json()["vectors"]
    assert {row["dtype"] for row in rows} == {"float32"}
    assert all(isinstance(row["vector"], bytes) for row in rows)


def test_enrolling_announces_itself_without_a_picture(client, system) -> None:
    enrol(client, system)
    announced = [p for _n, p in system.bus.replay(STUDENT_ENROLLED)]

    assert announced and announced[0].vectors > 1
    assert not hasattr(announced[0], "image")


# --- what the screen shows while the head turns ---------------------------


def test_the_sweep_reports_progress_and_coaching(client, system) -> None:
    _started, feedback, _done = enrol(client, system)

    assert feedback[0]["needed"] > 0
    assert feedback[-1]["collected"] >= feedback[0]["collected"]
    assert all(isinstance(f["reason"], str) and f["reason"] for f in feedback)
    assert set(feedback[-1]["angles_covered"]) <= {"left", "centre", "right"}


def test_no_face_in_front_of_the_camera_says_so(client, system) -> None:
    with_camera(system)
    system.enrolment.detector = NoFace()
    started = client.post("/api/enrol/start", json={
        "name": "Nikhil Rao", "roll_no": ROLL, "consent": {"granted_by": PARENT},
    }).json()

    refused = client.post(f"/api/enrol/{started['enrolment_id']}/frame")
    assert refused.status_code == 400
    assert "no face" in refused.json()["error"]


def test_an_abandoned_enrolment_is_dropped(client, system) -> None:
    """A half finished enrolment must not hold a child's vectors in memory
    until the robot is switched off."""
    with_camera(system)
    started = client.post("/api/enrol/start", json={
        "name": "Nikhil Rao", "roll_no": ROLL, "consent": {"granted_by": PARENT},
    }).json()

    system.clock.advance(system.cfg.enrolment.abandoned_after_seconds + 1)
    gone = client.post(f"/api/enrol/{started['enrolment_id']}/frame")

    assert gone.status_code == 400
    assert "timed out" in gone.json()["error"]


def test_a_school_can_remove_a_child_from_recognition(client, system) -> None:
    """A school that cannot delete a child's face data on request does not
    have consent, it has a form."""
    _started, _feedback, done = enrol(client, system)
    student_id = done.json()["student_id"]
    scope = system.orchestrator.scope

    removed = client.delete(f"/api/enrol/students/{student_id}").json()

    assert removed["removed"] == done.json()["vectors"]
    assert system.repos["embedding"].for_student(scope, student_id) == []
    assert not system.repos["consent"].is_granted(scope, student_id, "face_recognition")


def test_the_enrolled_list_shows_who_has_vectors(client, system) -> None:
    enrol(client, system)
    students = client.get("/api/enrol/students").json()["students"]

    enrolled = [s for s in students if s["vectors"]]
    assert len(enrolled) == 1
    assert enrolled[0]["consented"] is True
    assert len(students) == 6, "the seeded class plus the new child"


# --- live control ---------------------------------------------------------


class SlowVoice:
    """A speaker with something still to say. The null engine finishes
    instantly, so it cannot show whether pause interrupts or merely waits."""

    def __init__(self) -> None:
        self.handle: SpeechHandle | None = None

    def speak(self, text: str, language: str = "") -> SpeechHandle:
        self.handle = SpeechHandle(text=text, language=language)
        return self.handle

    def stop(self) -> None:
        if self.handle is not None and not self.handle.done:
            self.handle.cancel()

    def amplitude(self) -> float:
        return 0.0


def test_pause_stops_the_robot_mid_sentence() -> None:
    """A hard requirement, not a nicety. A teacher who has to wait out a
    paragraph before the room goes quiet stops using the pause button."""
    system = build()
    try:
        with TestClient(create_app(system)) as client:
            speaker = SlowVoice()
            system.voice.tts = speaker
            system.orchestrator.open_session()
            system.extras["machine"].state = SessionState.RUNNING

            saying = threading.Thread(
                target=system.bus.publish,
                args=(ROBOT_SAY, Utterance(text="A long sentence, still going.", language="en")),
                daemon=True,
            )
            saying.start()
            while speaker.handle is None:
                time.sleep(0.001)

            started = time.monotonic()
            client.post("/api/pause")
            saying.join(timeout=1.0)
            elapsed = time.monotonic() - started

            assert elapsed < 0.3, f"pause took {elapsed:.3f}s"
            assert speaker.handle.cancelled, "the robot was left talking"
            assert not saying.is_alive(), "the speaking thread never came back"
    finally:
        system.close()


def test_skip_moves_the_class_on(client, system) -> None:
    """The lesson is behind and the bell is in four minutes. The teacher
    knows that and the robot does not."""
    skipped: list[str] = []

    def on_step(_event, changed) -> None:
        if changed.step == "lesson" and not skipped:
            skipped.append(client.post("/api/skip").json()["skipped"])

    system.bus.subscribe("step.entered", on_step)
    assert system.orchestrator.run() is SessionState.CLOSED

    assert skipped == ["lesson"]
    segments = [p for _n, p in system.bus.replay(LESSON_SEGMENT)]
    assert len(segments) < 6, "the lesson ran to the end anyway"
    assert [p.step for _n, p in system.bus.replay("step.skipped")] == ["lesson"]


def test_the_teacher_can_silence_nudging_for_the_session(client, system) -> None:
    ctx = system.orchestrator.open_session()
    student = system.repos["student"].list_for_class(ctx.scope)[0]

    def drift() -> None:
        system.bus.publish(
            STUDENT_DISENGAGED,
            StudentDisengaged(track_id=1, student_id=student["id"], score=0.1,
                              drifting_for=9.0, at=1.0),
        )

    drift()
    assert [u for u in _said(system) if u.reason == "engagement"]

    client.post("/api/nudging", json={"enabled": False})
    before = len([u for u in _said(system) if u.reason == "engagement"])
    drift()
    assert len([u for u in _said(system) if u.reason == "engagement"]) == before


def test_the_nudging_switch_is_per_session_not_sticky(client, system) -> None:
    """A teacher silencing one lesson has not silenced the term."""
    system.orchestrator.open_session()
    client.post("/api/nudging", json={"enabled": False})
    system.orchestrator.close_session()

    ctx = system.orchestrator.open_session()
    student = system.repos["student"].list_for_class(ctx.scope)[0]
    system.bus.publish(
        STUDENT_DISENGAGED,
        StudentDisengaged(track_id=2, student_id=student["id"], score=0.1,
                          drifting_for=9.0, at=1.0),
    )

    assert [u for u in _said(system) if u.reason == "engagement"]


def test_tap_to_attribute_is_how_an_answer_gets_a_name(client, system) -> None:
    """Voice alone will not tell you which child spoke in a room of forty, so
    the person in the room decides."""
    ctx = system.orchestrator.open_session()
    student = system.repos["student"].list_for_class(ctx.scope)[1]

    client.post("/api/speaker", json={"student_id": student["id"]})
    assert client.post("/api/answer", json={"response": "it makes its own food"}).status_code == 200

    answered = [p for _n, p in system.bus.replay(QUIZ_ANSWERED)]
    assert answered[0].student_id == student["id"]
    assert answered[0].correct is None, "free text is left for the quizmaster to read"


def test_an_answer_with_nobody_tapped_is_refused(client, system) -> None:
    """Better a refusal the teacher can see than an answer recorded against
    the wrong child."""
    system.orchestrator.open_session()
    refused = client.post("/api/answer", json={"response": "the moon"})

    assert refused.status_code == 400
    assert "tap a name" in refused.json()["error"]


def test_a_relayed_question_carries_the_tapped_student(client, system) -> None:
    ctx = system.orchestrator.open_session()
    student = system.repos["student"].list_for_class(ctx.scope)[2]

    client.post("/api/speaker", json={"student_id": student["id"], "student_name": student["name"]})
    client.post("/api/ask", json={"text": "why are leaves green",
                                  "student_id": student["id"], "student_name": student["name"]})

    asked = [p for _n, p in system.bus.replay(QUESTION_ASKED)]
    assert asked[0].student_id == student["id"]


# --- the report -----------------------------------------------------------


def taught_and_answered(system):
    ctx = system.orchestrator.open_session()
    students = system.repos["student"].list_for_class(ctx.scope)
    system.repos["session"].mark_present(ctx.scope, ctx.session_id, students[0]["id"])
    system.repos["session"].mark_present(ctx.scope, ctx.session_id, students[1]["id"])

    lesson = system.content.load("en").lesson_for("photosynthesis")
    from lomas_core.contracts import LessonSegment

    for index in range(2):
        system.bus.publish(LESSON_SEGMENT, LessonSegment(
            session_id=ctx.session_id, lesson_id=lesson.id,
            segment_id=lesson.segments[index].id, index=index,
            total=len(lesson.segments), say=lesson.segments[index].say, display=""))

    system.bus.publish(QUESTION_ASKED, QuestionAsked(
        session_id=ctx.session_id, text="why are leaves green",
        student_id=students[0]["id"], student_name="Ananya"))

    system.repos["answer"].record(ctx.scope, ctx.session_id, students[0]["id"], "q1",
                                  "it makes food", True, 1800)
    system.orchestrator.close_session()
    return ctx


def test_the_report_reads_back_what_happened(client, system) -> None:
    ctx = taught_and_answered(system)
    body = client.get(f"/api/report/{ctx.session_id}").json()

    assert body["session"]["topic"] == "photosynthesis"
    assert body["attendance"]["count"] == 2
    lesson = system.content.load("en").lesson_for("photosynthesis")
    assert body["coverage"] == {
        "taught": 2,
        "total": len(lesson.segments),
        "segments": [s.id for s in lesson.segments[:2]],
    }
    assert body["questions"][0]["text"] == "why are leaves green"
    assert body["questions"][0]["asked_by"] == "Ananya"


def test_the_report_scores_per_student_in_roll_order(client, system) -> None:
    ctx = taught_and_answered(system)
    quiz = client.get(f"/api/report/{ctx.session_id}").json()["quiz"]

    assert [s["roll_no"] for s in quiz["students"]] == ["01", "02"]
    assert quiz["students"][0]["correct"] == 1
    assert quiz["students"][1]["answered"] == 0


def test_the_report_holds_no_judgement_of_any_child(client, system) -> None:
    """No engagement score, no ranking, nothing a parent could read as a
    verdict. A number saying how much a ten year old looked at a robot is not
    a fact about her, and in a document she takes home it becomes one."""
    ctx = taught_and_answered(system)
    body = json.dumps(client.get(f"/api/report/{ctx.session_id}").json()).lower()

    for word in ["attention", "engagement", "engaged", "drifting", "rank", "score", "percentile"]:
        assert word not in body, word


def test_attention_cannot_be_turned_on_in_a_report(system) -> None:
    assert system.cfg.teacher.report_shows_attention is False
    with pytest.raises(Exception):
        build("teacher.report_shows_attention=true")


def test_a_report_for_another_org_is_refused(client, system) -> None:
    ctx = taught_and_answered(system)
    system.cfg.tenancy.scratch_org_id = "somebody-else"

    refused = client.get(f"/api/report/{ctx.session_id}")
    assert refused.status_code == 400


def test_recent_sessions_are_listed_newest_first(client, system) -> None:
    taught_and_answered(system)
    system.clock.advance(60)
    taught_and_answered(system)

    sessions = client.get("/api/sessions").json()["sessions"]
    assert len(sessions) == 2
    assert sessions[0]["started_at"] >= sessions[1]["started_at"]


# --- what the teacher screen must not become ------------------------------


def test_the_teacher_screen_never_sorts_children_by_result() -> None:
    """A table sorted by score is a league table whatever the headings say."""
    code = "\n".join(l for l in TEACHER_JS.splitlines() if not l.strip().startswith("//"))
    assert "sort(" not in code
    assert "percent" not in code.lower()


def test_the_teacher_surface_is_served(client) -> None:
    assert client.get("/teacher/").status_code == 200


def test_the_surface_can_be_switched_off() -> None:
    system = build("teacher.enabled=false")
    try:
        with TestClient(create_app(system)) as client:
            assert system.enrolment is None
            assert client.get("/api/enrol/students").status_code == 404
            assert system.orchestrator.run() is SessionState.CLOSED
    finally:
        system.close()


def _said(system) -> list:
    return [p for _n, p in system.bus.replay(ROBOT_SAY)]
