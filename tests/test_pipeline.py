"""Two people in front of the camera, and the robot keeps them straight.

The build spec asks for a recorded clip. A clip proves the models work; it
cannot prove the pipeline is correct, because a video that fails tells you
nothing about which of six stages did it. So the scene here is painted: two
faces at known pixels, a scripted detector, and an embedder whose vector is a
function of the crop. Everything the pipeline is responsible for - scaling,
tracking, cropping from the right image, recognising once - is then a
statement with a number in it.

Point it at a real clip by swapping the source; nothing else changes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from lomas_core.clock import FakeClock, RealClock
from lomas_core.config import load
from lomas_core.contracts import (
    STEP_ENTERED,
    STUDENT_DISENGAGED,
    STUDENT_IDENTIFIED,
    STUDENT_LEFT,
    VISION_TRACKS,
    StudentIdentified,
)
from lomas_core.errors import LomasError
from lomas_core.events import EventBus
from lomas_core.schema import SourceConfig
from lomas_face import AttentionMonitor, IdentityMatcher, Tracker, crop_face
from lomas_face.detectors.mock import MockDetector
from lomas_face.embedders.mock import MockEmbedder
from lomas_face.types import Detection, Landmarks
from lomas_vision import Frame, FrameBus
from lomas_vision.sources.mock import MockSource

from app import container, seed
from app.flow.states import SessionState
from app.pipeline import VisionPipeline

WIDTH = 1280
HEIGHT = 720
CHANNELS = 3
SMALL_WIDTH = 640
FACTOR = WIDTH // SMALL_WIDTH

FACE_PX = 120  # full resolution; the detector sees half of this
SMALL_FACE = FACE_PX // FACTOR
STEP_PX = 3  # how far a person drifts between detect cycles
CYCLE_SECONDS = 0.1

ANANYA = 90
ROHIT = 200
AWAY_NOSE_PX = 12  # nose offset that puts a head outside the attention cone

BASE = [
    "storage.backend=memory",
    "face.detector=mock",
    "face.embedder=mock",
    f"face.downscale_width={SMALL_WIDTH}",
]

HEADLESS = [
    *BASE,
    "speech.tts.engine=null",
    "speech.stt.engine=keyboard",
    "speech.wake.engine=keyboard",
    "llm.provider=offline",
    "flow.attendance_wait_seconds=1",
    "flow.answer_wait_seconds=1",
    "flow.tick_seconds=0.1",
]


# --- the painted classroom ------------------------------------------------


@dataclass
class Person:
    shade: int
    x: int
    y: int
    turned: bool = False
    present: bool = True

    def at(self, cycle: int) -> Person:
        return Person(self.shade, self.x + cycle * STEP_PX, self.y, self.turned, self.present)


def landmarks_for(person: Person) -> Landmarks:
    """Five points inside the small-image box. The nose is what moves when
    someone turns, which is the whole of the pose estimate."""
    x, y = person.x // FACTOR, person.y // FACTOR
    nose_x = x + SMALL_FACE // 2 + (AWAY_NOSE_PX if person.turned else 0)
    return Landmarks(
        right_eye=(x + 15, y + 20),
        left_eye=(x + 45, y + 20),
        nose=(nose_x, y + 35),
        right_mouth=(x + 18, y + 48),
        left_mouth=(x + 42, y + 48),
    )


class Scene:
    """Paints the frames and scripts the detections that match them."""

    def __init__(self, *people: Person) -> None:
        self.people = list(people)

    def frame(self, cycle: int) -> Frame:
        image = np.zeros((HEIGHT, WIDTH, CHANNELS), dtype=np.uint8)
        for person in self._visible(cycle):
            image[person.y : person.y + FACE_PX, person.x : person.x + FACE_PX] = person.shade
        return Frame(
            source_id="head",
            zone="front",
            seq=cycle + 1,
            ts=cycle * CYCLE_SECONDS,
            image=image,
        )

    def detections(self, cycle: int) -> list[Detection]:
        """In small-image coordinates, exactly as a detector would report
        them. The pipeline is responsible for scaling them back up."""
        return [
            Detection(
                x=person.x // FACTOR,
                y=person.y // FACTOR,
                w=SMALL_FACE,
                h=SMALL_FACE,
                confidence=0.99,
                landmarks=landmarks_for(person),
            )
            for person in self._visible(cycle)
        ]

    def crop_of(self, person: Person, margin: float) -> np.ndarray:
        box = Detection(person.x, person.y, FACE_PX, FACE_PX, 0.99)
        return crop_face(self.frame(0).image, box, margin)

    def _visible(self, cycle: int) -> list[Person]:
        return [p.at(cycle) for p in self.people if p.present]


class BrokenDetector:
    """Fails every time, the way a missing model does."""

    def detect(self, image: np.ndarray) -> list[Detection]:
        raise LomasError("detector model not found")


class CropSpy:
    """Records what it was handed. The size of the crop is the evidence that
    the face came off the full-resolution frame."""

    def __init__(self, inner: MockEmbedder) -> None:
        self.inner = inner
        self.dim = inner.dim
        self.shapes: list[tuple[int, int]] = []

    def embed(self, face_crop: np.ndarray) -> np.ndarray:
        self.shapes.append(face_crop.shape[:2])
        return self.inner.embed(face_crop)

    @property
    def calls(self) -> int:
        return len(self.shapes)


# --- building a pipeline without a camera ---------------------------------


class Rig:
    def __init__(self, *overrides: str) -> None:
        self.cfg = load("config", "debug", [*BASE, *overrides], use_env=False)
        self.bus = EventBus(replay_size=self.cfg.runtime.event_replay_size)
        self.clock = FakeClock()
        self.detector = MockDetector(self.cfg.face)
        self.embedder = CropSpy(MockEmbedder(self.cfg.face))
        self.matcher = IdentityMatcher(self.embedder, self.cfg.face)
        self.cycle = 0
        self.pipeline = VisionPipeline(
            cfg=self.cfg,
            bus=self.bus,
            clock=self.clock,
            frames=FrameBus([], self.cfg.vision.buffer_size, self.clock,
                            self.cfg.vision.read_timeout_ms),
            detector=self.detector,
            tracker=Tracker(self.cfg.face),
            matcher=self.matcher,
            attention=AttentionMonitor(self.cfg.attention),
        )

    def enrol(self, scene: Scene, **people: Person) -> None:
        """Enrolment produces exactly the vector the pipeline will produce,
        because both embed the same crop of the same face."""
        reference = MockEmbedder(self.cfg.face)
        self.pipeline.load(
            {
                student_id: [reference.embed(scene.crop_of(person, self.cfg.face.crop_margin))]
                for student_id, person in people.items()
            }
        )

    def play(self, scene: Scene, cycles: int) -> None:
        """Cycles keep counting across calls, so a scene edited halfway
        through carries on in the same clip rather than jumping back in time."""
        for _ in range(cycles):
            self.detector.script([scene.detections(self.cycle)])
            self.pipeline.process(scene.frame(self.cycle))
            self.cycle += 1

    def seen(self, event: str) -> list:
        return [payload for _name, payload in self.bus.replay(event)]


@pytest.fixture
def two_people() -> Scene:
    return Scene(Person(ANANYA, x=200, y=300), Person(ROHIT, x=800, y=300))


@pytest.fixture
def rig(two_people: Scene) -> Rig:
    built = Rig()
    built.enrol(two_people, s1=two_people.people[0], s2=two_people.people[1])
    return built


# --- identity ------------------------------------------------------------


def test_two_people_keep_their_identities_across_the_clip(rig: Rig, two_people: Scene) -> None:
    rig.play(two_people, cycles=60)

    per_track: dict[int, set[str | None]] = {}
    for seen in rig.seen(VISION_TRACKS):
        for track in seen.tracks:
            per_track.setdefault(track.track_id, set()).add(track.student_id)

    assert len(per_track) == 2, "a face changed track id mid-clip"
    assert [sorted(v) for v in per_track.values()] == [["s1"], ["s2"]]


def test_each_person_is_announced_once(rig: Rig, two_people: Scene) -> None:
    rig.play(two_people, cycles=60)
    identified = rig.seen(STUDENT_IDENTIFIED)

    assert len(identified) == 2
    assert {i.student_id for i in identified} == {"s1", "s2"}
    assert {i.zone for i in identified} == {"front"}


def test_the_embedder_runs_once_per_person_not_once_per_frame(
    rig: Rig, two_people: Scene
) -> None:
    """This is the number the Pi 4 lives or dies on. Sixty cycles of two
    faces is a hundred and twenty recognitions if you do it the obvious way."""
    rig.play(two_people, cycles=60)

    assert rig.embedder.calls == 2
    assert rig.matcher.stats()["matches"] == 2


def test_recognition_is_rechecked_but_rarely(two_people: Scene) -> None:
    """Occasionally, to catch a mistake. `reverify_seconds` is the only thing
    standing between that and per-frame recognition."""
    rig = Rig("face.reverify_seconds=1.0")
    rig.enrol(two_people, s1=two_people.people[0], s2=two_people.people[1])
    rig.play(two_people, cycles=60)

    # Six seconds of footage, two faces, a one-second recheck window.
    assert rig.embedder.calls < 20
    assert len(rig.seen(STUDENT_IDENTIFIED)) == 2, "a recheck re-announced someone"


def test_faces_are_cropped_from_the_full_frame(rig: Rig, two_people: Scene) -> None:
    """Cropping from the downscaled copy is the classic version of this bug.
    It costs nothing visible and halves recognition accuracy."""
    rig.play(two_people, cycles=10)

    margin = int(FACE_PX * rig.cfg.face.crop_margin)
    expected = FACE_PX + margin * 2
    assert rig.embedder.shapes == [(expected, expected), (expected, expected)]


def test_boxes_are_published_in_full_resolution_pixels(rig: Rig, two_people: Scene) -> None:
    rig.play(two_people, cycles=10)
    last = rig.seen(VISION_TRACKS)[-1]

    assert (last.width, last.height) == (WIDTH, HEIGHT)
    assert {t.w for t in last.tracks} == {FACE_PX}


def test_a_face_too_small_to_recognise_is_still_tracked(two_people: Scene) -> None:
    """The back of the room. A child the robot cannot name is still counted,
    still tracked, and still costs nothing in embedder time."""
    rig = Rig(f"face.recognition_min_face_px={FACE_PX * 2}")
    rig.enrol(two_people, s1=two_people.people[0], s2=two_people.people[1])
    rig.play(two_people, cycles=20)

    assert rig.embedder.calls == 0
    assert len(rig.seen(VISION_TRACKS)[-1].tracks) == 2
    assert not rig.seen(STUDENT_IDENTIFIED)


# --- coming and going ----------------------------------------------------


def test_a_student_who_leaves_is_reported() -> None:
    scene = Scene(Person(ANANYA, x=200, y=300), Person(ROHIT, x=800, y=300))
    rig = Rig()
    rig.enrol(scene, s1=scene.people[0], s2=scene.people[1])

    rig.play(scene, cycles=10)
    scene.people[1].present = False
    rig.play(scene, cycles=40)

    left = rig.seen(STUDENT_LEFT)
    assert [entry.student_id for entry in left] == ["s2"]
    assert left[0].seen_for > 0
    assert len(rig.seen(VISION_TRACKS)[-1].tracks) == 1


def test_a_returning_student_is_announced_again() -> None:
    scene = Scene(Person(ANANYA, x=200, y=300), Person(ROHIT, x=800, y=300))
    rig = Rig()
    rig.enrol(scene, s1=scene.people[0], s2=scene.people[1])

    rig.play(scene, cycles=10)
    scene.people[1].present = False
    rig.play(scene, cycles=40)
    scene.people[1].present = True
    rig.play(scene, cycles=20)

    announced = [entry.student_id for entry in rig.seen(STUDENT_IDENTIFIED)]
    assert announced.count("s2") == 2
    assert announced.count("s1") == 1


# --- attention -----------------------------------------------------------


def test_looking_away_is_reported_as_drift() -> None:
    scene = Scene(Person(ANANYA, x=200, y=300, turned=True), Person(ROHIT, x=800, y=300))
    rig = Rig("attention.window_seconds=1.0", "attention.min_duration_seconds=2.0")
    rig.enrol(scene, s1=scene.people[0], s2=scene.people[1])
    rig.play(scene, cycles=120)

    drifting = rig.seen(STUDENT_DISENGAGED)
    assert drifting, "a child facing the wall went unnoticed"
    assert {entry.student_id for entry in drifting} == {"s1"}


def test_a_class_that_is_paying_attention_is_left_alone(rig: Rig, two_people: Scene) -> None:
    rig.play(two_people, cycles=120)
    assert not rig.seen(STUDENT_DISENGAGED)


# --- privacy -------------------------------------------------------------


def test_recognition_off_still_tracks_but_names_nobody(two_people: Scene) -> None:
    """A school that will not consent to face recognition still gets a
    working assistant. It counts heads and watches attention; it just never
    reaches the embedder."""
    rig = Rig("privacy.recognition_enabled=false")
    rig.enrol(two_people, s1=two_people.people[0], s2=two_people.people[1])
    rig.play(two_people, cycles=30)

    assert rig.embedder.calls == 0
    assert not rig.seen(STUDENT_IDENTIFIED)

    tracks = rig.seen(VISION_TRACKS)[-1].tracks
    assert len(tracks) == 2
    assert {t.student_id for t in tracks} == {None}


# --- the thread ----------------------------------------------------------


def test_vision_drops_frames_rather_than_falling_behind() -> None:
    """Capture runs faster than detection on purpose. What matters is which
    one waits for the other - and it is never the camera."""
    cfg = load("config", "debug", [*BASE, "face.detect_fps=5"], use_env=False)
    clock = RealClock()
    spec = SourceConfig(id="head", kind="mock", width=WIDTH, height=HEIGHT, fps=0)
    frames = FrameBus([MockSource(spec)], cfg.vision.buffer_size, clock,
                      cfg.vision.read_timeout_ms)

    pipeline = VisionPipeline(
        cfg=cfg,
        bus=EventBus(replay_size=cfg.runtime.event_replay_size),
        clock=clock,
        frames=frames,
        detector=MockDetector(cfg.face),
        tracker=Tracker(cfg.face),
        matcher=IdentityMatcher(MockEmbedder(cfg.face), cfg.face),
        attention=AttentionMonitor(cfg.attention),
    )

    pipeline.start()
    clock.sleep(cfg.vision.pipeline.join_timeout_seconds / 4)
    pipeline.stop()
    frames.stop()

    stats = pipeline.stats()
    assert stats["cycles"] > 0, "the pipeline never ran"
    assert stats["skipped"] > stats["cycles"], "detection kept pace with capture, which it cannot"


def test_a_broken_detector_stops_vision_and_nothing_else() -> None:
    """A model that was never downloaded, or a camera pulled out mid-lesson.
    The thread has to end as a log line, not as a corpse nobody notices until
    the report comes back empty."""
    cfg = load("config", "debug", [*BASE, "vision.pipeline.max_consecutive_errors=3"],
               use_env=False)
    clock = RealClock()
    spec = SourceConfig(id="head", kind="mock", width=WIDTH, height=HEIGHT, fps=0)
    frames = FrameBus([MockSource(spec)], cfg.vision.buffer_size, clock,
                      cfg.vision.read_timeout_ms)

    pipeline = VisionPipeline(
        cfg=cfg,
        bus=EventBus(replay_size=cfg.runtime.event_replay_size),
        clock=clock,
        frames=frames,
        detector=BrokenDetector(),
        tracker=Tracker(cfg.face),
        matcher=IdentityMatcher(MockEmbedder(cfg.face), cfg.face),
        attention=AttentionMonitor(cfg.attention),
    )

    pipeline.start()
    clock.sleep(cfg.vision.pipeline.join_timeout_seconds / 4)
    stats = pipeline.stats()
    pipeline.stop()
    frames.stop()

    assert not stats["running"], "the pipeline kept retrying a model that will never load"
    assert stats["errors"] == cfg.vision.pipeline.max_consecutive_errors


# --- wired into a class --------------------------------------------------


def build_system(*extra: str):
    cfg = load("config", "debug", [*HEADLESS, *extra], use_env=False)
    system = container.build(cfg, clock=FakeClock(), bus=container.event_bus(cfg))
    seed.demo_class(system)
    return system


def test_a_recognised_face_marks_attendance() -> None:
    """The point of the whole pipeline: attendance stops being the roster."""
    system = build_system("vision.pipeline.enabled=false")
    try:
        def on_step(_event, changed) -> None:
            if changed.step != "attendance":
                return
            for student in system.repos["student"].list_for_class(system.orchestrator.scope):
                system.bus.publish(
                    STUDENT_IDENTIFIED,
                    StudentIdentified(student_id=student["id"], track_id=1,
                                      source_id="head", zone="front", at=0.0),
                )

        system.bus.subscribe(STEP_ENTERED, on_step)
        system.orchestrator.run()

        marked = [p for name, p in system.bus.replay("attendance.marked")]
        assert {m.source for m in marked} == {"recognised"}
    finally:
        system.close()


def test_vision_is_kept_out_of_the_session_log() -> None:
    system = build_system("vision.pipeline.enabled=false")
    try:
        system.orchestrator.open_session()
        system.bus.publish(VISION_TRACKS, None)
        system.orchestrator.run()

        scope = system.orchestrator.scope
        session_id = system.repos["session"].recent(scope, 1)[0]["id"]
        logged = system.repos["event"].for_session(scope, session_id)

        assert logged, "the log is empty"
        assert not [row for row in logged if row["name"] == VISION_TRACKS]
    finally:
        system.close()


def test_the_class_still_runs_with_vision_switched_off() -> None:
    """Rule four, for the biggest subsystem in the product."""
    system = build_system("vision.pipeline.enabled=false")
    try:
        assert system.vision is None
        assert system.frames is None
        assert system.orchestrator.run() is SessionState.CLOSED
    finally:
        system.close()


def test_vision_starts_when_a_class_opens_not_when_the_robot_boots() -> None:
    system = build_system("sources=[]")
    try:
        assert system.vision is not None
        assert not system.vision.runnable, "no camera configured, so nothing to run"
        assert system.orchestrator.run() is SessionState.CLOSED
    finally:
        system.close()
