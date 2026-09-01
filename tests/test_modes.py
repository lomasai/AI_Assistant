"""Debug and user are two different machines built from one codebase.

The differences are small, deliberate, and each one is a decision somebody
could quietly undo. So each one is a test: sinks, error policy, which tenant
gets written to, and whether the overlay exists at all.
"""
from __future__ import annotations

import json
import logging

import pytest
from fastapi.testclient import TestClient

from lomas_core import logging as log
from lomas_core.clock import FakeClock
from lomas_core.config import load
from lomas_core.contracts import (
    QUESTION_ASKED,
    ROBOT_SAY,
    ROBOT_SPOKE,
    VISION_TRACKS,
    QuestionAsked,
    TracksSeen,
    TrackView,
    Utterance,
)
from lomas_core.events import EventBus
from lomas_llm.types import Completion, Message

from app import container, seed
from app.flow.states import SessionState
from app.observability.host import Host
from app.observability.metrics import LlmTap, Metrics

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


def build(mode: str, *extra: str):
    cfg = load("config", mode, [*HEADLESS, *extra], use_env=False)
    system = container.build(cfg, clock=FakeClock(), bus=container.event_bus(cfg))
    seed.demo_class(system)
    return system


@pytest.fixture(params=["debug", "user"])
def either(request):
    built = build(request.param)
    yield request.param, built
    built.close()


# --- the four differences -------------------------------------------------


def test_sinks_differ() -> None:
    debug = load("config", "debug", HEADLESS, use_env=False)
    user = load("config", "user", HEADLESS, use_env=False)

    assert debug.runtime.sinks == ["console", "jsonl"]
    assert user.runtime.sinks == ["console"]


def test_the_error_policy_differs() -> None:
    """On the bench a broken subscriber should stop everything. In a
    classroom it must not end the lesson in front of forty children."""
    debug = build("debug")
    user = build("user")
    try:
        assert debug.cfg.runtime.raise_on_handler_error is True
        assert user.cfg.runtime.raise_on_handler_error is False

        def explode(_event, _payload) -> None:
            raise RuntimeError("a bad subscriber")

        user.bus.subscribe(ROBOT_SAY, explode)
        user.bus.publish(ROBOT_SAY, Utterance(text="carry on", language="en"))

        debug.bus.subscribe(ROBOT_SAY, explode)
        with pytest.raises(RuntimeError):
            debug.bus.publish(ROBOT_SAY, Utterance(text="stop here", language="en"))
    finally:
        debug.close()
        user.close()


def test_the_target_tenant_differs() -> None:
    """Bench testing must never land in a real school's data."""
    debug = build("debug")
    user = build("user")
    try:
        assert debug.orchestrator.scope.org_id == debug.cfg.tenancy.scratch_org_id
        assert user.orchestrator.scope.org_id == user.cfg.tenancy.org_id
        assert debug.orchestrator.scope.org_id != user.orchestrator.scope.org_id
    finally:
        debug.close()
        user.close()


def test_the_overlay_is_not_served_in_user_mode() -> None:
    """Not hidden, not refused - absent. A route that only says no is still a
    route somebody can find."""
    user = build("user")
    try:
        with TestClient(_app(user)) as client:
            assert client.get("/debug/").status_code == 404
            assert client.get("/api/debug/metrics").status_code == 404
            assert client.get("/api/debug/config").status_code == 404
            assert user.metrics is None
    finally:
        user.close()


def test_the_overlay_is_served_in_debug_mode() -> None:
    debug = build("debug")
    try:
        with TestClient(_app(debug)) as client:
            assert client.get("/debug/").status_code == 200
            assert client.get("/api/debug/metrics").status_code == 200
            assert debug.metrics is not None
    finally:
        debug.close()


def test_the_class_runs_the_same_either_way(either) -> None:
    """The differences are diagnostics and tenancy. The lesson is the lesson."""
    _mode, system = either
    assert system.orchestrator.run() is SessionState.CLOSED

    sessions = system.repos["session"].recent(system.orchestrator.scope, 1)
    assert sessions[0]["status"] == "closed"


# --- what the overlay shows -----------------------------------------------


@pytest.fixture
def debug_system():
    built = build("debug")
    yield built
    built.close()


@pytest.fixture
def client(debug_system):
    with TestClient(_app(debug_system)) as opened:
        yield opened


def test_metrics_counts_and_times_events(client, debug_system) -> None:
    debug_system.orchestrator.open_session()
    debug_system.bus.publish(ROBOT_SAY, Utterance(text="Leaves make food.", language="en"))
    debug_system.bus.publish(ROBOT_SPOKE, Utterance(text="Leaves make food.", language="en"))

    body = client.get("/api/debug/metrics").json()

    assert body["events"]["counts"][ROBOT_SAY] >= 1
    assert body["latency"]["speak"]["count"] == 1
    assert body["latency"]["speak"]["last_ms"] is not None


def test_the_track_table_comes_from_what_vision_published(client, debug_system) -> None:
    """Not a second copy of the tracker. Two answers to who is in the room is
    one answer too many."""
    debug_system.bus.publish(VISION_TRACKS, TracksSeen(
        source_id="head", zone="front", at=1.0, width=1280, height=720,
        tracks=(TrackView(track_id=3, x=10, y=20, w=120, h=120, student_id="s1",
                          attention=0.8, yaw=4.0, pitch=1.0, seen_for=12.0),),
    ))

    tracks = client.get("/api/debug/metrics").json()["tracks"]
    assert [t["track_id"] for t in tracks] == [3]
    assert tracks[0]["student_id"] == "s1"


def test_the_noisy_event_is_counted_but_not_listed(client, debug_system) -> None:
    """Ten a second would push everything else out of the tail."""
    for _ in range(5):
        debug_system.bus.publish(VISION_TRACKS, TracksSeen(
            source_id="head", zone="front", at=1.0, width=1280, height=720))
    debug_system.bus.publish(ROBOT_SAY, Utterance(text="still visible", language="en"))

    events = client.get("/api/debug/events").json()
    assert events["counts"][VISION_TRACKS] == 5
    assert VISION_TRACKS not in [entry["event"] for entry in events["recent"]]
    assert ROBOT_SAY in [entry["event"] for entry in events["recent"]]


def test_the_prompt_and_tokens_are_visible(client, debug_system) -> None:
    """Half of debugging a model is reading what you actually sent it."""
    debug_system.orchestrator.open_session()
    debug_system.bus.publish(
        QUESTION_ASKED,
        QuestionAsked(session_id=debug_system.orchestrator.ctx.session_id,
                      text="why are leaves green"),
    )

    llm = client.get("/api/debug/metrics").json()["llm"]
    assert "tutor" in llm["agents"]
    assert llm["calls"], "no model call was recorded"

    call = llm["calls"][0]
    assert any("leaves green" in message["content"] for message in call["prompt"])
    assert call["provider"] == "offline"
    assert call["ms"] >= 0


def test_an_unpriced_model_costs_nothing_rather_than_a_guess(client, debug_system) -> None:
    """A made-up cost is worse than an absent one."""
    debug_system.orchestrator.open_session()
    debug_system.bus.publish(
        QUESTION_ASKED,
        QuestionAsked(session_id=debug_system.orchestrator.ctx.session_id, text="why"),
    )

    llm = client.get("/api/debug/metrics").json()["llm"]
    assert llm["cost"] == 0
    assert llm["input_tokens"] >= 0


def test_a_priced_model_is_costed_from_config() -> None:
    system = build("debug", 'debug.cost_per_million={"offline":[3.0,15.0]}')
    try:
        with TestClient(_app(system)) as client:
            system.orchestrator.open_session()
            system.bus.publish(
                QUESTION_ASKED,
                QuestionAsked(session_id=system.orchestrator.ctx.session_id,
                              text="why are leaves green"),
            )
            llm = client.get("/api/debug/metrics").json()["llm"]
            assert llm["cost"] > 0
    finally:
        system.close()


def test_the_tap_never_changes_an_answer() -> None:
    """A tap that can alter a reply is not a tap."""
    class Fixed:
        name = "fixed"

        def complete(self, messages, **options):
            return Completion(text="unchanged", provider=self.name)

        def stream(self, messages, **options):
            yield "unchanged"

    tapped = LlmTap(Fixed(), keep=5)
    answer = tapped.complete([Message("user", "anything")])

    assert answer.text == "unchanged"
    assert answer.provider == "fixed"
    assert len(tapped.calls) == 1


def test_plugins_shows_what_was_chosen_next_to_what_exists(client) -> None:
    body = client.get("/api/debug/plugins").json()

    assert body["chosen"]["llm"] == "offline"
    assert "offline" in body["available"]["llm"]
    assert len(body["available"]["tts"]) > 1


def test_the_resolved_config_is_readable(client, debug_system) -> None:
    """Half of debugging this system is finding out which layer won."""
    body = client.get("/api/debug/config").json()

    assert body["runtime"]["mode"] == "debug"
    assert body["storage"]["backend"] == "memory", "the --set override should be visible"


def test_nothing_in_the_debug_api_changes_anything(client, debug_system) -> None:
    """Diagnostics are subscribers, never participants."""
    for path in ["/api/debug/metrics", "/api/debug/plugins", "/api/debug/config"]:
        assert client.post(path).status_code == 405


# --- the host panel -------------------------------------------------------


def test_host_readings_degrade_instead_of_failing() -> None:
    """None of this exists on Windows and all of it exists on the Pi. A
    diagnostics panel that raises on a laptop is a panel nobody develops."""
    reading = Host().snapshot()

    assert set(reading) == {"cores", "temperature_c", "throttled", "memory", "load", "uptime_s"}
    assert isinstance(reading["cores"], list)
    assert isinstance(reading["throttled"], list)
    assert reading["temperature_c"] is None or reading["temperature_c"] > 0


def test_the_first_cpu_reading_is_empty_and_the_second_is_not() -> None:
    """Busy share is a difference of two samples, so the first call has
    nothing to compare against and says so rather than reporting zero."""
    host = Host()
    first = host.cpu()
    second = host.cpu()

    assert first == []
    assert all(0.0 <= value <= 100.0 for value in second)


# --- the tap is not built in a classroom ----------------------------------


def test_user_mode_holds_no_prompt_in_memory() -> None:
    """The tap keeps every prompt the robot has sent. That is right on a
    bench and wrong in a school."""
    user = build("user")
    try:
        user.orchestrator.open_session()
        user.bus.publish(
            QUESTION_ASKED,
            QuestionAsked(session_id=user.orchestrator.ctx.session_id, text="why"),
        )

        tutor = next(a for a in user.agents.agents if a.name == "tutor")
        assert not isinstance(tutor.deps.llm, LlmTap)
        assert user.metrics is None
    finally:
        user.close()


def test_metrics_can_be_switched_off_in_debug_too() -> None:
    system = build("debug", "debug.enabled=false")
    try:
        assert system.metrics is None
        with TestClient(_app(system)) as client:
            assert client.get("/api/debug/metrics").status_code == 404
        assert system.orchestrator.run() is SessionState.CLOSED
    finally:
        system.close()


# --- log formatting -------------------------------------------------------


def test_debug_lines_carry_machinery_and_user_lines_do_not() -> None:
    record = logging.LogRecord("lomas.flow", logging.INFO, __file__, 1, "-> lesson", None, None)

    assert log.HumanFormatter().format(record) == "-> lesson"
    detailed = log.DebugFormatter().format(record)
    assert "flow" in detailed and "-> lesson" in detailed and detailed != "-> lesson"

    machine = json.loads(log.JsonlFormatter().format(record))
    assert machine["logger"] == "lomas.flow"
    assert machine["message"] == "-> lesson"


def _app(system):
    from app.web.server import create_app

    return create_app(system)


# --- what a half-built robot does -----------------------------------------
#
# All three of these were found on a real Pi, and all three had the same
# shape: something the robot is better with, but can teach without, ended the
# lesson instead.


def test_a_profile_that_says_user_gets_user_behaviour() -> None:
    """Profiles do not inherit from each other. pi.yaml is built on
    default.yaml, not on user.yaml, so it said `mode: user` and then raised
    on a bad subscriber - ending a class on the actual robot."""
    for name in ["pi", "user"]:
        cfg = load("config", name, [], use_env=False)
        assert cfg.runtime.mode == "user"
        assert cfg.runtime.raise_on_handler_error is False, name

    for name in ["debug", "demo"]:
        cfg = load("config", name, [], use_env=False)
        assert cfg.runtime.raise_on_handler_error is True, name


def test_the_error_policy_can_still_be_overridden() -> None:
    """Derived, not dictated. Someone may want debug diagnostics with a
    classroom's forgiveness, or the reverse."""
    cfg = load("config", "pi", ["runtime.raise_on_handler_error=true"], use_env=False)
    assert cfg.runtime.raise_on_handler_error is True


def test_a_robot_with_no_voice_still_teaches() -> None:
    """piper not on PATH used to propagate out of the bus and kill the class
    thread mid-greeting."""
    from lomas_core.errors import LomasError

    system = build("user")
    try:
        class Mute:
            def speak(self, text, language=""):
                raise LomasError("'piper' is not on PATH")

            def stop(self):
                ...

            def amplitude(self):
                return 0.0

        system.voice.tts = Mute()
        assert system.orchestrator.run() is SessionState.CLOSED

        sessions = system.repos["session"].recent(system.orchestrator.scope, 1)
        assert sessions[0]["status"] == "closed"
    finally:
        system.close()


def test_no_recognition_is_not_no_vision() -> None:
    """onnxruntime missing used to fail five detect cycles in a row and stop
    vision entirely - losing the boxes, the attention and the head tracking
    along with the names."""
    import numpy as np
    from lomas_core.errors import LomasError
    from lomas_face import IdentityMatcher
    from lomas_face.types import Detection, Track

    cfg = load("config", "debug", HEADLESS, use_env=False)

    class Missing:
        dim = 512

        def embed(self, crop):
            raise LomasError("onnxruntime is not installed")

    matcher = IdentityMatcher(Missing(), cfg.face)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    box = Detection(x=100, y=100, w=200, h=200, confidence=0.9)
    track = Track(track_id=1, box=box, first_seen=0.0, last_seen=0.0)

    assert matcher.resolve(track, frame, 1.0) is None
    assert matcher.unavailable, "it should remember and stop trying"

    # And it must not keep raising on every later frame.
    for tick in range(10):
        assert matcher.resolve(track, frame, 2.0 + tick) is None

    assert matcher.embed_calls == 1, "it kept calling a model that cannot load"
    assert matcher.stats()["recognition"] == "off"
