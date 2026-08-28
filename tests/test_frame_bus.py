"""Three cameras at once, with no hardware and no cross-contamination."""
from __future__ import annotations

import itertools
import time

import numpy as np
import pytest

from lomas_core.clock import RealClock
from lomas_core.schema import SourceConfig
from lomas_vision import CAMERA_SOURCES, FrameBus, build_sources, crop_rectangle
from lomas_vision.source import rotate

FRAMES = 100
SMALL = 64
UNPACED = 0


def spec(source_id: str, **overrides) -> SourceConfig:
    base = dict(id=source_id, kind="mock", width=SMALL, height=SMALL, fps=UNPACED)
    base.update(overrides)
    return SourceConfig(**base)


@pytest.fixture
def bus():
    sources = build_sources([spec("head"), spec("left", zone="left"), spec("right", zone="right")])
    running = FrameBus(sources, buffer_size=8, clock=RealClock(), read_timeout_ms=200)
    running.start()
    yield running
    running.stop()


def test_every_backend_is_registered():
    assert set(CAMERA_SOURCES.keys()) == {
        "picamera2", "usb", "rtsp", "file", "folder", "mock"
    }


def test_picamera2_imports_on_a_laptop():
    """The module must import off-device; it may only complain on open()."""
    source = CAMERA_SOURCES.create("picamera2", spec("pi", kind="picamera2"))
    assert source.source_id == "pi"


def test_sequence_is_monotonic_per_source(bus):
    seen: dict[str, list[int]] = {}
    for frame in itertools.islice(bus.subscribe(), FRAMES):
        seen.setdefault(frame.source_id, []).append(frame.seq)

    assert seen, "no frames arrived"
    for source_id, seqs in seen.items():
        assert seqs == sorted(seqs), f"{source_id} delivered frames out of order"
        assert len(set(seqs)) == len(seqs), f"{source_id} repeated a sequence number"


def test_subscription_to_one_source_never_sees_another(bus):
    frames = list(itertools.islice(bus.subscribe("left"), 25))
    assert frames
    assert {f.source_id for f in frames} == {"left"}
    assert {f.zone for f in frames} == {"left"}


def test_zone_travels_with_the_frame(bus):
    frame = next(iter(bus.subscribe("right")))
    assert frame.zone == "right"


def test_latest_is_per_source(bus):
    list(itertools.islice(bus.subscribe(), 30))
    for source_id in ("head", "left", "right"):
        frame = bus.latest(source_id)
        assert frame is not None
        assert frame.source_id == source_id


def test_frames_are_distinct(bus):
    frames = list(itertools.islice(bus.subscribe("head"), 5))
    checksums = {int(f.image.sum()) for f in frames}
    assert len(checksums) > 1, "mock source produced identical frames"


def test_slow_consumer_drops_instead_of_stalling():
    sources = build_sources([spec("head")])
    bus = FrameBus(sources, buffer_size=2, clock=RealClock(), read_timeout_ms=200)
    bus.start()
    try:
        stream = bus.subscribe("head")
        first = next(stream)

        # Let capture run far ahead of this consumer. Bounded so a stalled
        # capture thread fails the test instead of hanging the suite.
        target = first.seq + 200
        deadline = time.monotonic() + 5.0
        while bus.stats()["head"]["captured"] < target:
            if time.monotonic() > deadline:
                pytest.fail("capture stalled waiting for the consumer")
            time.sleep(0.001)

        assert bus.stats()["head"]["dropped"] > 0
    finally:
        bus.stop()


def test_stats_report_per_source(bus):
    list(itertools.islice(bus.subscribe(), 20))
    stats = bus.stats()
    assert set(stats) == {"head", "left", "right"}
    assert all(s["captured"] > 0 for s in stats.values())


def test_disabled_sources_are_not_built():
    built = build_sources([spec("on"), spec("off", enabled=False)])
    assert [s.source_id for s in built] == ["on"]


def test_rotation_swaps_axes():
    image = np.zeros((10, 20, 3), dtype=np.uint8)
    assert rotate(image, 0).shape == (10, 20, 3)
    assert rotate(image, 90).shape == (20, 10, 3)
    assert rotate(image, 180).shape == (10, 20, 3)


def test_zoom_crops_the_centre():
    full = crop_rectangle(1.0, 3280, 2464, 1.0, 2.5)
    assert full == (0, 0, 3280, 2464)

    x, y, w, h = crop_rectangle(2.0, 3280, 2464, 1.0, 2.5)
    assert (w, h) == (1640, 1232)
    assert x == (3280 - w) // 2 and y == (2464 - h) // 2


def test_zoom_is_clamped():
    assert crop_rectangle(9.0, 3280, 2464, 1.0, 2.5) == crop_rectangle(2.5, 3280, 2464, 1.0, 2.5)
