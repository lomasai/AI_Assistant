"""A timeline of one run, for reading on another machine.

The point of this file is that we stop guessing at performance. Two rules
matter more than anything it records: it must be off unless asked for, and it
must never become the slowness it was written to find.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from lomas_core.clock import FakeClock
from lomas_core.config import load
from lomas_core.contracts import QUESTION_ASKED, ROBOT_SAY, VISION_TRACKS, QuestionAsked, Utterance

from app import container, seed
from app.flow.states import SessionState
from app.observability.trace import Trace, build_trace, trace_path

HEADLESS = [
    "storage.backend=memory",
    "web.enabled=false",
    "hardware.enabled=false",
    "vision.pipeline.enabled=false",
    "speech.tts.engine=null",
    "speech.stt.engine=keyboard",
    "speech.wake.engine=keyboard",
    "llm.provider=offline",
    "flow.attendance_wait_seconds=0.2",
    "flow.answer_wait_seconds=0.2",
    "flow.tick_seconds=0.02",
    "flow.stage_timeout_seconds.lesson=1",
    "flow.stage_timeout_seconds.interaction=0.4",
    "flow.stage_timeout_seconds.quiz=1",
]


def build(tmp_path: Path, *extra: str):
    cfg = load(
        "config", "debug",
        [*HEADLESS, f"trace.directory={tmp_path.as_posix()}", *extra],
        use_env=False,
    )
    system = container.build(cfg, clock=FakeClock(), bus=container.event_bus(cfg))
    seed.demo_class(system)
    return system


def records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# --- off unless asked for -------------------------------------------------


def test_nothing_is_written_unless_it_is_switched_on(tmp_path: Path) -> None:
    """A class in a school writes no trace. This is a bench tool that happens
    to run on the robot, not a thing every lesson pays for."""
    system = build(tmp_path)
    try:
        assert system.trace is None
        assert system.orchestrator.run() is SessionState.CLOSED
        assert list(tmp_path.glob("*.jsonl")) == []
    finally:
        system.close()


def test_it_is_off_unless_somebody_is_going_to_read_it() -> None:
    """Measuring costs something, so nothing measures by default - except
    the robot being worked on, where a run that leaves nothing behind makes
    "why was it silent?" unanswerable. The runtime log is not committed;
    this is what `sync` actually pushes."""
    for mode in ["debug", "user", "demo"]:
        assert load("config", mode, [], use_env=False).trace.enabled is False, mode

    assert load("config", "pi", [], use_env=False).trace.enabled is True


# --- what a run leaves behind ---------------------------------------------


@pytest.fixture
def traced(tmp_path: Path):
    system = build(tmp_path, "trace.enabled=true", "trace.sample_seconds=0")
    yield system
    if system.trace is not None:
        system.close()


def test_a_run_leaves_a_timeline_and_a_summary(traced) -> None:
    path = traced.trace.path
    traced.orchestrator.run()
    traced.close()

    assert path.exists()
    assert path.with_suffix(".md").exists(), "the file a person reads first"

    written = records(path)
    kinds = {r["kind"] for r in written}
    assert kinds <= {"meta", "event", "sample", "span"}
    assert written[0]["phase"] == "start"
    assert written[-1]["phase"] == "stop"


def test_every_record_carries_when_it_happened(traced) -> None:
    """Seconds since the run began. Latency is a subtraction afterwards, which
    is why almost nothing is computed while recording."""
    traced.orchestrator.run()
    path = traced.trace.path
    traced.close()

    stamps = [r["t"] for r in records(path)]
    assert stamps == sorted(stamps), "the timeline is out of order"
    assert stamps[0] >= 0


def test_the_settings_that_change_speed_are_recorded(traced) -> None:
    """Two traces are only comparable if you know what was different. Enough
    to tell runs apart, not the whole config."""
    path = traced.trace.path
    traced.close()

    config = records(path)[0]["config"]
    assert set(config) >= {"detect_fps", "downscale_width", "detector", "embedder",
                           "camera", "mjpeg_fps", "llm", "tts", "stt"}


def test_the_events_a_lesson_produces_are_all_there(traced) -> None:
    traced.orchestrator.run()
    path = traced.trace.path
    traced.close()

    seen = {r["event"] for r in records(path) if r["kind"] == "event"}
    assert {"session.opened", "step.entered", "lesson.segment",
            "robot.say", "robot.spoke", "session.closed"} <= seen


def test_a_span_records_a_duration(traced) -> None:
    traced.trace.span("http", 0.125, {"path": "/api/state"})
    path = traced.trace.path
    traced.close()

    spans = [r for r in records(path) if r["kind"] == "span"]
    assert spans[0]["ms"] == 125.0
    assert spans[0]["path"] == "/api/state"


# --- it must not become the slowness --------------------------------------


def test_a_flood_is_sampled_not_recorded_in_full(tmp_path: Path) -> None:
    """Vision publishes eight times a second. Every line of that would bury
    the handful that explain a slow answer."""
    system = build(tmp_path, "trace.enabled=true", "trace.sample_seconds=0",
                   'trace.sample_every={"robot.say":5}')
    path = system.trace.path
    try:
        for index in range(20):
            system.bus.publish(ROBOT_SAY, Utterance(text=f"line {index}", language="en"))
    finally:
        system.close()

    kept = [r for r in records(path) if r.get("event") == ROBOT_SAY]
    assert len(kept) == 4, "one in five, not all twenty"

    counted = records(path)[-1]["counts"][ROBOT_SAY]
    assert counted == 20, "sampling must not distort the count"


def test_a_large_payload_can_be_left_out(tmp_path: Path) -> None:
    """Timing is the point. A tracks payload is long and adds nothing to it."""
    system = build(tmp_path, "trace.enabled=true", "trace.sample_seconds=0",
                   "trace.sample_every={}")
    path = system.trace.path
    try:
        system.bus.publish(VISION_TRACKS, {"tracks": [{"x": 1}] * 50, "width": 1280})
    finally:
        system.close()

    line = next(r for r in records(path) if r.get("event") == VISION_TRACKS)
    assert "payload" not in line, "it was excluded in config and still written"


def test_a_full_queue_drops_records_rather_than_blocking(tmp_path: Path) -> None:
    """The rule that matters. A profiler that waits on a slow SD card is a
    profiler measuring itself."""
    cfg = load("config", "debug",
               [*HEADLESS, f"trace.directory={tmp_path.as_posix()}",
                "trace.enabled=true", "trace.sample_seconds=0", "trace.queue_size=16"],
               use_env=False)
    trace = Trace(cfg, container.event_bus(cfg), FakeClock(), tmp_path / "full.jsonl")
    trace._queue.maxsize = 16

    for index in range(500):
        trace._record("event", {"event": f"n{index}"})

    assert trace.dropped > 0
    assert trace._queue.qsize() <= 16


def test_the_whole_class_still_runs_with_tracing_on(tmp_path: Path) -> None:
    system = build(tmp_path, "trace.enabled=true")
    try:
        assert system.orchestrator.run() is SessionState.CLOSED
        sessions = system.repos["session"].recent(system.orchestrator.scope, 1)
        assert sessions[0]["status"] == "closed"
    finally:
        system.close()


# --- the name says which run it was ---------------------------------------


def test_the_filename_sorts_and_identifies(tmp_path: Path) -> None:
    """`2026-09-08_1435_pi_a3f2.jsonl` - sortable, and says which run without
    opening it. Two runs a minute apart must not collide."""
    cfg = load("config", "pi", [f"trace.directory={tmp_path.as_posix()}"], use_env=False)

    first = trace_path(cfg, FakeClock())
    second = trace_path(cfg, FakeClock())

    assert first.suffix == ".jsonl"
    assert "_pi_" in first.name, "the profile, not the mode it resolves to"
    assert first != second, "two runs in the same minute overwrote each other"


def test_build_returns_nothing_when_disabled(tmp_path: Path) -> None:
    cfg = load("config", "debug", [f"trace.directory={tmp_path.as_posix()}"], use_env=False)
    assert build_trace(cfg, container.event_bus(cfg), FakeClock()) is None


# --- the summary a person reads first -------------------------------------


def test_the_summary_names_the_busiest_events(traced) -> None:
    traced.orchestrator.run()
    path = traced.trace.path
    traced.close()

    summary = path.with_suffix(".md").read_text(encoding="utf-8")
    assert "Busiest events" in summary
    assert "robot.say" in summary
    assert "per second" in summary


def test_a_question_and_its_answer_are_both_timed(traced) -> None:
    """The pair that answers "why is it slow to reply"."""
    ctx = traced.orchestrator.open_session()
    traced.bus.publish(QUESTION_ASKED, QuestionAsked(session_id=ctx.session_id,
                                                     text="why are leaves green"))
    path = traced.trace.path
    traced.close()

    events = [r["event"] for r in records(path) if r["kind"] == "event"]
    assert events.index("question.asked") < events.index("question.answered")
