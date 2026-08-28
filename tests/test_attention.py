"""Attention scoring, and the limits that stop it becoming nagging."""
from __future__ import annotations

import pytest

from lomas_core.schema import AttentionConfig, FaceConfig, PoseConfig
from lomas_face import AttentionMonitor, Detection, Landmarks, Pose, Track, estimate_pose

TICK = 0.5
FACE = 80


def cfg(**overrides) -> AttentionConfig:
    base = dict(
        window_seconds=2.0,
        threshold=0.45,
        min_duration_seconds=3.0,
        cooldown_seconds=60.0,
        max_nudges_per_session=3,
        cone_yaw_degrees=35.0,
        cone_pitch_degrees=25.0,
    )
    base.update(overrides)
    return AttentionConfig(**base)


def make_track(track_id: int = 1, yaw: float = 0.0, student: str | None = "s1") -> Track:
    box = Detection(x=0, y=0, w=FACE, h=FACE, confidence=0.9)
    return Track(
        track_id=track_id,
        box=box,
        first_seen=0.0,
        last_seen=0.0,
        student_id=student,
        pose=Pose(yaw=yaw, pitch=0.0, roll=0.0),
    )


def run(monitor: AttentionMonitor, track: Track, yaw: float, seconds: float, start: float = 0.0):
    """Hold a head angle for a while and collect anything the monitor says."""
    signals = []
    steps = int(seconds / TICK)
    for step in range(steps):
        track.pose = Pose(yaw=yaw, pitch=0.0, roll=0.0)
        signals.extend(monitor.update([track], ts=start + step * TICK))
    return signals


def test_facing_forward_stays_engaged():
    monitor = AttentionMonitor(cfg())
    track = make_track()
    assert run(monitor, track, yaw=0.0, seconds=20.0) == []
    assert track.attention > 0.9


def test_looking_away_long_enough_fires_once():
    monitor = AttentionMonitor(cfg())
    track = make_track()
    signals = run(monitor, track, yaw=80.0, seconds=20.0)

    assert len(signals) == 1, f"expected exactly one nudge, got {len(signals)}"
    assert signals[0].student_id == "s1"
    assert signals[0].drifting_for >= 3.0


def test_a_brief_glance_away_does_not_fire():
    monitor = AttentionMonitor(cfg())
    track = make_track()
    assert run(monitor, track, yaw=80.0, seconds=1.0) == []


def test_cooldown_suppresses_a_second_nudge():
    monitor = AttentionMonitor(cfg(cooldown_seconds=60.0))
    track = make_track()

    first = run(monitor, track, yaw=80.0, seconds=20.0, start=0.0)
    second = run(monitor, track, yaw=80.0, seconds=20.0, start=25.0)

    assert len(first) == 1
    assert second == [], "fired again inside the cooldown"


def test_nudging_resumes_after_the_cooldown():
    monitor = AttentionMonitor(cfg(cooldown_seconds=10.0))
    track = make_track()

    # First window is shorter than the cooldown, so it can only fire once.
    first = run(monitor, track, yaw=80.0, seconds=8.0, start=0.0)
    later = run(monitor, track, yaw=80.0, seconds=8.0, start=100.0)

    assert len(first) == 1
    assert len(later) == 1


def test_session_budget_is_enforced():
    monitor = AttentionMonitor(cfg(cooldown_seconds=1.0, max_nudges_per_session=2))
    track = make_track()

    fired = 0
    for window in range(6):
        fired += len(run(monitor, track, yaw=80.0, seconds=12.0, start=window * 50.0))

    assert fired == 2, f"budget of 2 was exceeded: {fired}"


def test_reset_restores_the_budget():
    monitor = AttentionMonitor(cfg(cooldown_seconds=1.0, max_nudges_per_session=1))
    track = make_track()
    assert len(run(monitor, track, yaw=80.0, seconds=12.0)) == 1

    monitor.reset()
    track = make_track()
    assert len(run(monitor, track, yaw=80.0, seconds=12.0, start=200.0)) == 1


def test_two_students_have_separate_cooldowns():
    monitor = AttentionMonitor(cfg(max_nudges_per_session=5))
    a = make_track(track_id=1, student="s1")
    b = make_track(track_id=2, student="s2")

    signals = []
    for step in range(40):
        a.pose = b.pose = Pose(yaw=80.0, pitch=0.0, roll=0.0)
        signals.extend(monitor.update([a, b], ts=step * TICK))

    assert {s.student_id for s in signals} == {"s1", "s2"}


def test_disabled_monitor_is_silent():
    monitor = AttentionMonitor(cfg(enabled=False))
    assert run(monitor, make_track(), yaw=80.0, seconds=30.0) == []


def test_missing_pose_holds_the_score():
    """No landmarks means no opinion - a child who turns fully away stops
    being detected, and that is a departure, not inattention."""
    monitor = AttentionMonitor(cfg())
    track = make_track()

    for step in range(10):
        track.pose = None
        monitor.update([track], ts=step * TICK)

    assert track.attention == pytest.approx(1.0)


def test_pose_reads_a_turned_head():
    pose_cfg = PoseConfig()
    forward = Landmarks(
        right_eye=(40.0, 50.0), left_eye=(80.0, 50.0), nose=(60.0, 70.0),
        right_mouth=(45.0, 90.0), left_mouth=(75.0, 90.0),
    )
    turned = Landmarks(
        right_eye=(40.0, 50.0), left_eye=(80.0, 50.0), nose=(78.0, 70.0),
        right_mouth=(45.0, 90.0), left_mouth=(75.0, 90.0),
    )

    straight = estimate_pose(forward, pose_cfg)
    sideways = estimate_pose(turned, pose_cfg)

    assert abs(straight.yaw) < 5.0
    assert sideways.yaw > straight.yaw + 20.0


def test_pose_returns_nothing_for_degenerate_landmarks():
    flat = Landmarks(
        right_eye=(50.0, 50.0), left_eye=(50.0, 50.0), nose=(50.0, 50.0),
        right_mouth=(50.0, 50.0), left_mouth=(50.0, 50.0),
    )
    assert estimate_pose(flat, PoseConfig()) is None


def test_face_config_defaults_are_sane():
    face = FaceConfig()
    assert face.detect_fps < 15, "detection should run slower than capture"
    assert face.min_face_px < 80, "detection floor sits below the recognition floor"
