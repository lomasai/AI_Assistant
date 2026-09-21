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
