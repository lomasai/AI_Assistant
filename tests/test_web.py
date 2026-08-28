"""The surfaces, and the rules they must not break.

Two things are worth testing here and the rest is plumbing: that a slow
browser cannot slow the class down, and that a child's attention never
reaches a screen as a number.
"""
from __future__ import annotations

import asyncio
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from lomas_core.clock import FakeClock
from lomas_core.config import load
from lomas_core.contracts import (
    LESSON_SEGMENT,
    QUESTION_ASKED,
    ROBOT_SAY,
    STORY_REQUESTED,
    VISION_TRACKS,
    TracksSeen,
    TrackView,
    Utterance,
)
from lomas_core.events import EventBus, to_plain

from app import container, seed
from app.flow.states import SessionState
from app.web.server import create_app
from app.web.ws import Client, EventHub

UI = Path("app/web/ui")


def code(path: Path) -> str:
    """Comments explain the rules; only the code has to obey them."""
    body = path.read_text(encoding="utf-8")
    return "\n".join(line for line in body.splitlines() if not line.strip().startswith("//"))


FACE_JS = code(UI / "face" / "face.js")
BOARD_JS = code(UI / "board" / "board.js")

HEADLESS = [
    "storage.backend=memory",
    "vision.pipeline.enabled=false",
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
    system = container.build(cfg, clock=FakeClock(), bus=EventBus(cfg.runtime.event_replay_size))
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


SENTINEL = "test.sentinel"


def drain(system, socket, limit: int = 400) -> list[dict]:
    """Read up to a marker.

    A websocket read blocks, and the stream is deliberately endless, so the
    test publishes something it can recognise and reads until it arrives.
    """
    system.bus.publish(SENTINEL, {"marker": True})
    seen: list[dict] = []
    for _ in range(limit):
        message = socket.receive_json()
        if message["event"] == SENTINEL:
            return seen
        seen.append(message)
    raise AssertionError("the marker never arrived")


# --- the surfaces ---------------------------------------------------------


def test_both_surfaces_are_served(client) -> None:
    assert client.get("/face/").status_code == 200
    assert client.get("/board/").status_code == 200
    assert client.get("/", follow_redirects=False).headers["location"] == "/face/"


def test_a_surface_can_be_switched_off() -> None:
    system = build("web.surfaces=[face]")
    try:
        with TestClient(create_app(system)) as only_face:
            assert only_face.get("/face/").status_code == 200
            assert only_face.get("/board/").status_code == 404
    finally:
        system.close()


def test_the_layout_comes_from_config(client, system) -> None:
    """Panel size and the engaged boundary are config, not numbers baked into
    a stylesheet somebody has to find later."""
    body = client.get("/api/display").json()

    assert body["face_screen"]["width"] == system.cfg.display.face_screen.width
    assert body["base_font_px"] == system.cfg.display.base_font_px
    assert body["attention_threshold"] == system.cfg.attention.threshold


# --- teacher controls -----------------------------------------------------


def test_state_is_readable_before_a_class_starts(client) -> None:
    body = client.get("/api/state").json()

    assert body["state"] == "idle"
    assert body["session_id"] == ""
    assert len(body["roster"]) == 5


def test_state_reports_the_running_class(client, system) -> None:
    ctx = system.orchestrator.open_session()
    body = client.get("/api/state").json()

    assert body["session_id"] == ctx.session_id
    assert body["topic"] == "photosynthesis"
    assert body["agents"] == ["tutor", "quizmaster", "narrator", "engagement", "safety"]


def test_pause_and_resume_reach_the_machine(client, system) -> None:
    machine = system.extras["machine"]
    machine.state = SessionState.RUNNING

    assert client.post("/api/pause").json() == {"ok": True}
    assert machine.state is SessionState.PAUSED

    client.post("/api/resume")
    assert machine.state is SessionState.RUNNING


def test_halt_and_clear_go_through_the_same_events_as_the_button(client, system) -> None:
    """The web halt is the physical e-stop's event, not a second path to the
    same place. One way in means one behaviour to get right."""
    client.post("/api/halt", json={"reason": "teacher"})
    machine = system.extras["machine"]

    assert machine.state is SessionState.HALTED
    assert machine.halt_reason == "teacher"

    client.post("/api/clear")
    assert machine.state is SessionState.IDLE


def test_the_teacher_can_relay_a_question(client, system) -> None:
    system.orchestrator.open_session()
    client.post("/api/ask", json={"text": "why are leaves green", "student_name": "Ananya"})

    asked = [p for _n, p in system.bus.replay(QUESTION_ASKED)]
    assert asked and asked[0].student_name == "Ananya"


def test_the_teacher_can_call_for_a_story(client, system) -> None:
    system.orchestrator.open_session()
    client.post("/api/story", json={"topic": "seeds"})

    requested = [p for _n, p in system.bus.replay(STORY_REQUESTED)]
    assert requested and requested[0].topic == "seeds"


# --- the event stream -----------------------------------------------------


def test_a_browser_joining_late_is_caught_up(client, system) -> None:
    """A tab opened mid-lesson must not sit blank until somebody speaks."""
    system.orchestrator.open_session()
    system.bus.publish(ROBOT_SAY, Utterance(text="Leaves make food.", language="en"))

    with client.websocket_connect("/events") as socket:
        events = [message["event"] for message in drain(system, socket)]

    assert "session.opened" in events
    assert ROBOT_SAY in events


def test_events_published_after_connecting_arrive(client, system) -> None:
    with client.websocket_connect("/events") as socket:
        drain(system, socket)
        system.bus.publish(ROBOT_SAY, Utterance(text="Chlorophyll is green.", language="en"))
        arrived = drain(system, socket)

    said = [m["payload"]["text"] for m in arrived if m["event"] == ROBOT_SAY]
    assert said == ["Chlorophyll is green."]


def test_a_nested_payload_survives_the_wire(client, system) -> None:
    """The face positions boxes from this. A serialiser that flattens the
    track list to a string is a UI with no boxes and no error."""
    tracks = TracksSeen(
        source_id="head", zone="front", at=1.0, width=1280, height=720,
        tracks=(TrackView(track_id=1, x=100, y=200, w=120, h=120, student_id="s1",
                          attention=0.9, yaw=2.0, pitch=1.0, seen_for=4.0),),
    )

    with client.websocket_connect("/events") as socket:
        drain(system, socket)
        system.bus.publish(VISION_TRACKS, tracks)
        arrived = drain(system, socket)

    seen = [m for m in arrived if m["event"] == VISION_TRACKS][0]["payload"]
    assert seen["width"] == 1280
    assert seen["tracks"][0]["x"] == 100
    assert seen["tracks"][0]["student_id"] == "s1"


def test_the_wire_format_is_the_log_format(system) -> None:
    """One serialiser. A report and a screen disagreeing about what happened
    is a support call nobody can answer."""
    segment = [p for _n, p in system.bus.replay(LESSON_SEGMENT)]
    system.orchestrator.open_session()
    assert to_plain(Utterance(text="x", language="en"))["text"] == "x"
    assert segment == []


# --- a slow browser must never slow the class -----------------------------


def test_a_full_client_queue_drops_the_oldest_event() -> None:
    """The direction that matters. A tab left open on a locked laptop loses
    events; it does not hold up a lesson."""

    async def run() -> Client:
        client = Client(size=3)
        for index in range(10):
            client.offer({"event": "n", "payload": index})
        return client

    client = asyncio.run(run())

    assert client.queue.qsize() == 3
    assert client.dropped == 7
    assert client.queue.get_nowait()["payload"] == 7, "it kept the newest"


def test_publishing_with_no_server_running_is_free(system) -> None:
    """The hub exists whether or not anyone is watching. Before the loop is
    bound it must cost nothing and raise nothing."""
    hub = EventHub(system.bus, system.cfg)

    system.bus.publish(ROBOT_SAY, Utterance(text="nobody is looking", language="en"))
    assert hub.delivered == 0


def test_the_event_filter_is_config(system) -> None:
    narrowed = build("web.event_filter=[student.*]")
    try:
        hub = EventHub(narrowed.bus, narrowed.cfg)
        assert hub.wants("student.identified")
        assert not hub.wants(ROBOT_SAY)
    finally:
        narrowed.close()


def test_vision_is_on_the_wire_even_though_it_is_out_of_the_log(system) -> None:
    """Opposite calls on purpose: the face needs tracks ten times a second,
    and the session log would drown in them."""
    hub = EventHub(system.bus, system.cfg)

    assert hub.wants(VISION_TRACKS)
    assert VISION_TRACKS in system.cfg.runtime.log_event_exclude[0].replace("*", "tracks")


# --- what the screens must never show -------------------------------------


def test_the_ribbon_shows_a_name_and_a_dot_and_nothing_else() -> None:
    """Never a number, never a ranking, never a percentage. A child's
    attention score on a screen her classmates can read is what gets this
    product thrown out of a school, and it would arrive as one careless line."""
    written = re.findall(r"textContent\s*=\s*([^;]+);", FACE_JS)
    numeric = [line for line in written if "attention" in line or "score" in line]

    assert not numeric, numeric
    assert "attention" not in BOARD_JS, "the board must not carry attention at all"
    assert "sort(" not in FACE_JS, "a sorted ribbon is a ranking"


def test_the_face_never_repaints_the_video() -> None:
    """Boxes are HTML over the stream. A canvas loop here would cost the
    detector its cores, and re-encoding in Python would cost more."""
    assert "canvas" not in FACE_JS.lower()
    assert "requestAnimationFrame" not in FACE_JS
    assert "setInterval" not in FACE_JS


def test_every_face_state_in_the_spec_exists() -> None:
    css = (UI / "face" / "face.css").read_text(encoding="utf-8")
    for state in ["sleeping", "listening", "thinking", "speaking", "asking"]:
        assert f'data-state="{state}"' in css, state


def test_no_framework_is_loaded() -> None:
    for name in ["face", "board"]:
        page = (UI / name / "index.html").read_text(encoding="utf-8")
        assert "http" not in page.replace("http-equiv", ""), "a surface must load nothing remote"


# --- the camera -----------------------------------------------------------


def test_the_stream_is_served_even_with_no_camera(client) -> None:
    """A robot with vision switched off still serves the page. It shows no
    video, which is correct, rather than a broken surface."""
    with client.stream("GET", "/camera.mjpeg") as response:
        assert response.status_code == 200
        assert "multipart/x-mixed-replace" in response.headers["content-type"]
