"""Tracking is a state machine, so it is tested as one - no model, no images."""
from __future__ import annotations

import pytest

from lomas_core.schema import FaceConfig
from lomas_face import DETECTORS, Detection, Tracker, iou

FACE = 80
CONFIDENT = 0.9
STEP = 6


def cfg(**overrides) -> FaceConfig:
    base = dict(track_birth_hits=1, track_iou_threshold=0.35, track_death_seconds=1.5)
    base.update(overrides)
    return FaceConfig(**base)


def face_at(x: int, y: int, size: int = FACE, confidence: float = CONFIDENT) -> Detection:
    return Detection(x=x, y=y, w=size, h=size, confidence=confidence)


def test_iou_of_identical_boxes_is_one():
    box = face_at(0, 0)
    assert iou(box, box) == pytest.approx(1.0)


def test_iou_of_disjoint_boxes_is_zero():
    assert iou(face_at(0, 0), face_at(500, 500)) == 0.0


def test_one_face_moving_keeps_its_id():
    tracker = Tracker(cfg())
    ids = set()
    for step in range(30):
        tracks = tracker.update([face_at(100 + step * STEP, 100)], ts=step * 0.1)
        assert len(tracks) == 1
        ids.add(tracks[0].track_id)
    assert len(ids) == 1, f"track id changed mid-flight: {ids}"


def test_two_faces_crossing_do_not_swap_ids():
    """They approach, pass at different heights, and separate - which is what
    two children leaning across a desk actually look like."""
    tracker = Tracker(cfg())
    seen: dict[int, list[int]] = {}

    for step in range(40):
        left_x = 60 + step * STEP
        right_x = 400 - step * STEP
        tracks = tracker.update(
            [face_at(left_x, 100), face_at(right_x, 220)], ts=step * 0.1
        )
        for track in tracks:
            seen.setdefault(track.track_id, []).append(track.box.y)

    assert len(seen) == 2, f"expected exactly two tracks, got {len(seen)}"
    for track_id, heights in seen.items():
        assert len(set(heights)) == 1, f"track {track_id} jumped rows - ids were swapped"


def test_track_needs_birth_hits_before_it_is_confirmed():
    tracker = Tracker(cfg(track_birth_hits=3))
    assert tracker.update([face_at(100, 100)], ts=0.0) == []
    assert tracker.update([face_at(102, 100)], ts=0.1) == []
    assert len(tracker.update([face_at(104, 100)], ts=0.2)) == 1


def test_track_dies_after_the_death_window():
    tracker = Tracker(cfg(track_death_seconds=1.0))
    tracker.update([face_at(100, 100)], ts=0.0)
    assert len(tracker.active()) == 1

    tracker.update([], ts=0.5)
    assert len(tracker.active()) == 1, "must survive a brief miss"

    tracker.update([], ts=1.2)
    assert tracker.active() == []


def test_reappearing_face_gets_a_new_id():
    tracker = Tracker(cfg(track_death_seconds=1.0))
    first = tracker.update([face_at(100, 100)], ts=0.0)[0].track_id
    tracker.update([], ts=2.0)
    second = tracker.update([face_at(100, 100)], ts=2.1)[0].track_id
    assert first != second


def test_low_confidence_detections_are_ignored():
    tracker = Tracker(cfg(min_confidence=0.6))
    assert tracker.update([face_at(100, 100, confidence=0.2)], ts=0.0) == []


def test_faces_below_the_size_floor_are_ignored():
    tracker = Tracker(cfg(min_face_px=40))
    assert tracker.update([face_at(100, 100, size=20)], ts=0.0) == []


def test_max_tracks_is_respected():
    tracker = Tracker(cfg(max_tracks=2))
    crowd = [face_at(i * 200, 100) for i in range(5)]
    assert len(tracker.update(crowd, ts=0.0)) == 2


def test_identity_rides_along_once_attached():
    """This is the property the whole recognition design depends on."""
    tracker = Tracker(cfg())
    track = tracker.update([face_at(100, 100)], ts=0.0)[0]
    track.student_id = "student-7"

    for step in range(1, 20):
        tracks = tracker.update([face_at(100 + step * STEP, 100)], ts=step * 0.1)
        assert tracks[0].student_id == "student-7"


def test_mock_detector_is_registered_and_scriptable():
    detector = DETECTORS.create("mock", cfg())
    detector.script([[face_at(10, 10)], [], [face_at(12, 10)]])
    assert len(detector.detect()) == 1
    assert detector.detect() == []
    assert len(detector.detect()) == 1
    assert detector.detect() == [], "runs off the end of the script harmlessly"
    assert detector.calls == 4


def test_every_detector_is_registered():
    assert set(DETECTORS.keys()) == {"yunet", "mediapipe", "mock"}
