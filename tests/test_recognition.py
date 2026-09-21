"""Straightening a face before it is recognised.

SFace was trained on faces with the eyes in the same place every time, and
was being handed whatever the detector's box happened to contain. The detector
already finds the eyes, so the only real requirement is that enrolment and
recognition agree - a vector stored from a plain crop and compared against a
straightened one is a child the robot does not know.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lomas_core.config import load
from lomas_core.schema import EnrolmentConfig, FaceConfig, PoseConfig
from lomas_face import EMBEDDERS, EnrolmentSession
from lomas_face.align import as_row, face_for
from lomas_face.types import Detection, Landmarks

MODEL = Path("models/face_recognition_sface_2021dec.onnx")


def a_face(x: int = 40, y: int = 30, w: int = 120, h: int = 120, marks: bool = True) -> Detection:
    landmarks = Landmarks(
        right_eye=(x + w * 0.3, y + h * 0.35),
        left_eye=(x + w * 0.7, y + h * 0.35),
        nose=(x + w * 0.5, y + h * 0.55),
        right_mouth=(x + w * 0.35, y + h * 0.75),
        left_mouth=(x + w * 0.65, y + h * 0.75),
    ) if marks else None
    return Detection(x=x, y=y, w=w, h=h, confidence=0.95, landmarks=landmarks)


class Fake:
    """An embedder that records what it was handed."""

    dim = 4

    def __init__(self, aligns: bool = True) -> None:
        self.aligns = aligns
        self.rows: list[np.ndarray] = []
        self.crops: list[np.ndarray] = []

    def align(self, image, row):
        self.rows.append(row)
        return np.ones((112, 112, 3), dtype=np.uint8) if self.aligns else None

    def embed(self, face_crop):
        self.crops.append(face_crop)
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


def a_frame() -> np.ndarray:
    rng = np.random.default_rng(7)
    return (rng.random((320, 320, 3)) * 255).astype(np.uint8)


# --- the row the aligner wants --------------------------------------------


def test_the_detector_already_found_everything_alignment_needs() -> None:
    row = as_row(a_face())

    assert row is not None
    assert row.shape == (1, 15), "box, five points, score"
    assert list(row[0][:4]) == [40, 30, 120, 120]


def test_a_face_with_no_landmarks_cannot_be_straightened() -> None:
    assert as_row(a_face(marks=False)) is None


# --- which picture gets embedded ------------------------------------------


def test_a_face_is_straightened_when_it_can_be() -> None:
    embedder = Fake()

    picture = face_for(embedder, a_frame(), a_face(), margin=0.2, align=True)

    assert embedder.rows, "the aligner was never asked"
    assert picture.shape == (112, 112, 3)


def test_alignment_is_off_by_config() -> None:
    embedder = Fake()

    picture = face_for(embedder, a_frame(), a_face(), margin=0.2, align=False)

    assert not embedder.rows
    assert picture.shape[0] != 112, "a plain crop, as before"


def test_a_face_the_aligner_refuses_still_gets_recognised() -> None:
    """At the edge of the frame, or on an older OpenCV. Losing the name is
    worse than a slightly worse crop."""
    embedder = Fake(aligns=False)

    picture = face_for(embedder, a_frame(), a_face(), margin=0.2, align=True)

    assert picture.size > 0


def test_an_embedder_that_cannot_align_is_not_asked_to() -> None:
    class Plain:
        dim = 4

        def embed(self, face_crop):
            return np.zeros(4, dtype=np.float32)

    picture = face_for(Plain(), a_frame(), a_face(), margin=0.2, align=True)
    assert picture.size > 0


# --- enrolment and recognition have to agree ------------------------------


def test_enrolment_stores_the_same_kind_of_picture_recognition_compares() -> None:
    embedder = Fake()
    sweep = EnrolmentSession(embedder, EnrolmentConfig(), PoseConfig(), align=True)

    sweep.add_frame(a_frame(), a_face(w=140, h=140))

    assert embedder.rows, "enrolment took the plain crop while matching took the aligned one"


def test_the_two_paths_are_one_function() -> None:
    """Belt and braces: if either stops calling it, they can drift apart
    without a single test failing on its own."""
    import inspect

    from lomas_face import enrolment, identity

    assert "face_for" in inspect.getsource(enrolment.EnrolmentSession.add_frame)
    assert "face_for" in inspect.getsource(identity.IdentityMatcher.resolve)


def test_the_pi_profile_straightens_faces() -> None:
    assert load("config", "pi", [], use_env=False).face.align is True


# --- against the real model -----------------------------------------------


def test_straightening_a_face_with_the_real_model() -> None:
    if not MODEL.exists():
        pytest.skip("run python tools/fetch_models.py to test against the real model")

    embedder = EMBEDDERS.create("sface", FaceConfig(embedder_model_path=str(MODEL)))
    frame = a_frame()

    straightened = face_for(embedder, frame, a_face(), margin=0.2, align=True)

    assert straightened.shape == (112, 112, 3), "SFace's own alignment size"
    assert embedder.embed(straightened).shape == (128,)


# --- which of them is talking ---------------------------------------------


def moving_mouth(frame: np.ndarray, face: Detection, by: int) -> np.ndarray:
    """The same frame with the mouth region changed, as a face mid-syllable
    differs from the frame before it."""
    from lomas_face.mouth import mouth_box
    from lomas_face.types import Track

    x, y, w, h = mouth_box(Track(track_id=1, box=face, first_seen=0.0, last_seen=0.0))
    changed = frame.copy()
    changed[y: y + h, x: x + w] = np.clip(
        changed[y: y + h, x: x + w].astype(int) + by, 0, 255).astype(np.uint8)
    return changed


def a_track(track_id: int, face: Detection):
    from lomas_face.types import Track

    return Track(track_id=track_id, box=face, first_seen=0.0, last_seen=1.0)


def test_a_mouth_that_moves_scores_higher_than_one_that_does_not() -> None:
    from lomas_face.mouth import Mouths

    talker, quiet = a_face(x=20, y=20), a_face(x=180, y=20)
    frame = a_frame()
    mouths = Mouths(smoothing=0.0, size=24)

    mouths.update(frame, [a_track(1, talker), a_track(2, quiet)])
    moved = mouths.update(moving_mouth(frame, talker, by=60),
                          [a_track(1, talker), a_track(2, quiet)])

    assert moved[1] > moved[2] * 3, "the one whose mouth moved is not standing out"
    assert moved[2] < 0.01


def test_the_first_frame_of_a_face_scores_nothing() -> None:
    """There is nothing to compare it against, and a made-up number here
    would name whoever happened to walk in."""
    from lomas_face.mouth import Mouths

    mouths = Mouths(smoothing=0.5, size=24)
    assert mouths.update(a_frame(), [a_track(1, a_face())]) == {}


def test_a_face_that_leaves_is_forgotten() -> None:
    from lomas_face.mouth import Mouths

    face = a_face()
    mouths = Mouths(smoothing=0.0, size=24)
    mouths.update(a_frame(), [a_track(1, face)])
    mouths.update(moving_mouth(a_frame(), face, by=40), [a_track(1, face)])
    assert mouths.score(1) > 0

    mouths.update(a_frame(), [])
    assert mouths.score(1) == 0.0, "a score left behind would answer for somebody who left"


def test_a_face_with_no_landmarks_has_no_mouth_to_watch() -> None:
    from lomas_face.mouth import Mouths, mouth_box

    assert mouth_box(a_track(1, a_face(marks=False))) is None
    assert Mouths(smoothing=0.5, size=24).update(a_frame(), [a_track(1, a_face(marks=False))]) == {}
