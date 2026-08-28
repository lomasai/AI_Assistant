"""Recognition runs once per track, not once per frame.

The count in test_recognises_once_per_track is the number the whole design
turns on. If it ever reads 100, the Pi will not cope and the fix is here,
not in the hardware.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from lomas_core.schema import EnrolmentConfig, FaceConfig, PoseConfig, PrivacyConfig
from lomas_face import (
    EMBEDDERS,
    Detection,
    EnrolmentSession,
    IdentityMatcher,
    Landmarks,
    Tracker,
    crop_face,
    distance,
)

FRAMES = 100
FACE_PX = 120
FRAME_W = 1280
FRAME_H = 720
STEP = 2


def cfg(**overrides) -> FaceConfig:
    base = dict(
        embedder="mock",
        embedding_dim=64,
        match_threshold=0.5,
        reverify_seconds=20.0,
        unknown_after_attempts=3,
        recognition_min_face_px=80,
        track_birth_hits=1,
    )
    base.update(overrides)
    return FaceConfig(**base)


def frame_for(person: int) -> np.ndarray:
    """A whole frame filled with one person's number.

    The mock embedder seeds from the mean of the crop, so a fill value stands
    in for an identity and any crop of this frame embeds identically.
    """
    return np.full((FRAME_H, FRAME_W, 3), person, dtype=np.uint8)


def face_at(x: int, y: int = 200, size: int = FACE_PX) -> Detection:
    return Detection(x=x, y=y, w=size, h=size, confidence=0.95)


def embedder(config: FaceConfig):
    return EMBEDDERS.create(config.embedder, config)


def enrol(config: FaceConfig, people: list[int]) -> tuple[IdentityMatcher, object]:
    model = embedder(config)
    matcher = IdentityMatcher(model, config)
    matcher.load(
        {f"student-{p}": [model.embed(crop_face(frame_for(p), face_at(100), config.crop_margin))]
         for p in people}
    )
    matcher.reset()
    return matcher, model


def test_every_embedder_is_registered():
    assert set(EMBEDDERS.keys()) == {"arcface_onnx", "mock"}


def test_recognises_once_per_track():
    """One stable track across a hundred frames costs one embedding."""
    config = cfg()
    matcher, _ = enrol(config, [7])
    tracker = Tracker(config)
    frame = frame_for(7)

    for step in range(FRAMES):
        ts = step * 0.1  # 10 seconds total, under reverify_seconds
        tracks = tracker.update([face_at(100 + step * STEP)], ts)
        for track in tracks:
            matcher.resolve(track, frame, ts)

    assert matcher.stats()["embed_calls"] == 1, (
        f"embedder ran {matcher.stats()['embed_calls']} times for one track. "
        "Identity must be resolved on new tracks only."
    )
    assert tracker.active()[0].student_id == "student-7"


def test_identity_survives_the_whole_track():
    config = cfg()
    matcher, _ = enrol(config, [3])
    tracker = Tracker(config)
    frame = frame_for(3)

    names = set()
    for step in range(50):
        ts = step * 0.1
        for track in tracker.update([face_at(100 + step * STEP)], ts):
            matcher.resolve(track, frame, ts)
            names.add(track.student_id)

    assert names == {"student-3"}


def test_reverify_runs_again_after_the_window():
    config = cfg(reverify_seconds=5.0)
    matcher, _ = enrol(config, [4])
    tracker = Tracker(config)
    frame = frame_for(4)

    for step in range(30):
        ts = step * 1.0  # 30 seconds, so reverify fires several times
        for track in tracker.update([face_at(100)], ts):
            matcher.resolve(track, frame, ts)

    calls = matcher.stats()["embed_calls"]
    assert 1 < calls <= 7, f"expected a handful of re-checks, got {calls}"


def test_a_new_track_costs_one_more_embedding():
    config = cfg(track_death_seconds=1.0)
    matcher, _ = enrol(config, [5])
    tracker = Tracker(config)
    frame = frame_for(5)

    for track in tracker.update([face_at(100)], 0.0):
        matcher.resolve(track, frame, 0.0)
    tracker.update([], 5.0)
    for track in tracker.update([face_at(100)], 5.1):
        matcher.resolve(track, frame, 5.1)

    assert matcher.stats()["embed_calls"] == 2


def test_unknown_face_returns_no_name():
    config = cfg()
    matcher, _ = enrol(config, [7])
    tracker = Tracker(config)
    stranger = frame_for(200)

    for track in tracker.update([face_at(100)], 0.0):
        assert matcher.resolve(track, stranger, 0.0) is None
    assert matcher.stats()["unknowns"] == 1


def test_a_stranger_stops_costing_cpu():
    """Without this, an unrecognised visitor burns the embedder forever."""
    config = cfg(unknown_after_attempts=3, reverify_seconds=0.001)
    matcher, _ = enrol(config, [7])
    tracker = Tracker(config)
    stranger = frame_for(200)

    for step in range(FRAMES):
        ts = step * 0.1
        for track in tracker.update([face_at(100)], ts):
            matcher.resolve(track, stranger, ts)

    assert matcher.stats()["embed_calls"] == 3, "gave up after the configured attempts"


def test_faces_too_small_are_not_embedded():
    config = cfg(recognition_min_face_px=80, min_face_px=10)
    matcher, _ = enrol(config, [7])
    tracker = Tracker(config)
    frame = frame_for(7)

    for track in tracker.update([face_at(100, size=40)], 0.0):
        matcher.resolve(track, frame, 0.0)

    assert matcher.stats()["embed_calls"] == 0
    assert matcher.stats()["skipped_too_small"] == 1


def test_two_people_get_different_names():
    config = cfg()
    matcher, _ = enrol(config, [11, 22])
    tracker = Tracker(config)

    a = tracker.update([face_at(100)], 0.0)[0]
    matcher.resolve(a, frame_for(11), 0.0)
    b = tracker.update([face_at(100), face_at(600)], 0.1)[-1]
    matcher.resolve(b, frame_for(22), 0.1)

    assert a.student_id == "student-11"
    assert b.student_id == "student-22"


def test_distance_is_zero_for_the_same_vector():
    config = cfg()
    model = embedder(config)
    vector = model.embed(np.full((10, 10, 3), 42, dtype=np.uint8))
    assert distance(vector, vector) == pytest.approx(0.0, abs=1e-6)


# --- enrolment -------------------------------------------------------------


def sweep_cfg(**overrides) -> EnrolmentConfig:
    base = dict(keep_best=2, min_quality=0.0, min_face_px=96, angle_yaw_degrees=20.0)
    base.update(overrides)
    return EnrolmentConfig(**base)


def looking(yaw_offset: float) -> Landmarks:
    """Landmarks whose nose is displaced to fake a head turn."""
    return Landmarks(
        right_eye=(140.0, 250.0),
        left_eye=(220.0, 250.0),
        nose=(180.0 + yaw_offset, 290.0),
        right_mouth=(150.0, 330.0),
        left_mouth=(210.0, 330.0),
    )


def detection_with(landmarks: Landmarks) -> Detection:
    return Detection(x=120, y=200, w=FACE_PX, h=FACE_PX, confidence=0.95, landmarks=landmarks)


def test_enrolment_collects_three_angles():
    config = cfg()
    session = EnrolmentSession(embedder(config), sweep_cfg(), PoseConfig())

    for offset in (-30.0, -25.0, 0.0, 2.0, 25.0, 30.0):
        feedback = session.add_frame(frame_for(9), detection_with(looking(offset)))
        assert feedback.accepted, feedback.reason

    assert set(session.angles_covered) == {"left", "centre", "right"}
    assert session.complete

    result = session.finish()
    assert result.dim == 64
    assert len(result.variants) == 3
    assert result.angles == ["centre", "left", "right"]


def test_enrolment_returns_vectors_and_nothing_else():
    """The privacy promise is a property of the type, not a policy.

    Every array leaving enrolment must be one-dimensional. An image would be
    two or three, so this fails the moment someone starts carrying pictures
    out of the sweep.
    """
    config = cfg()
    session = EnrolmentSession(embedder(config), sweep_cfg(), PoseConfig())
    session.add_frame(frame_for(9), detection_with(looking(0.0)))
    result = session.finish()

    carried = [getattr(result, f.name) for f in dataclasses.fields(result)]
    arrays = [v for v in carried if isinstance(v, np.ndarray)]
    arrays += [v for group in carried if isinstance(group, list) for v in group
               if isinstance(v, np.ndarray)]

    assert arrays, "expected at least the mean vector"
    for array in arrays:
        assert array.ndim == 1, "a 2D or 3D array here would be an image"

    # And the session itself must not be holding frames after the fact.
    retained = [v for v in vars(session).values() if isinstance(v, np.ndarray)]
    assert retained == [], "enrolment kept an image"


def test_enrolment_result_is_ready_for_the_repository():
    config = cfg()
    session = EnrolmentSession(embedder(config), sweep_cfg(), PoseConfig())
    for offset in (-30.0, 0.0, 30.0):
        session.add_frame(frame_for(9), detection_with(looking(offset)))

    rows = session.finish().as_rows()
    assert rows[0][2] == "float32"
    assert all(isinstance(row[0], bytes) for row in rows)
    assert {row[4] for row in rows} == {"mean", "left", "centre", "right"}


def test_small_faces_are_refused_with_guidance():
    config = cfg()
    session = EnrolmentSession(embedder(config), sweep_cfg(min_face_px=200), PoseConfig())
    feedback = session.add_frame(frame_for(9), detection_with(looking(0.0)))

    assert not feedback.accepted
    assert "closer" in feedback.reason


def test_finishing_with_nothing_is_an_error():
    config = cfg()
    session = EnrolmentSession(embedder(config), sweep_cfg(), PoseConfig())
    with pytest.raises(Exception, match="no usable frames"):
        session.finish()


def test_store_images_cannot_be_turned_on():
    """Not a bool. The config must not be able to promise what the code and
    the schema deliberately cannot do."""
    with pytest.raises(Exception):
        PrivacyConfig(store_images=True)

    assert PrivacyConfig().store_images is False
