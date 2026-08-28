from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np

from lomas_core import logging as log
from lomas_core.clock import Clock
from lomas_core.contracts import (
    STUDENT_DISENGAGED,
    STUDENT_IDENTIFIED,
    STUDENT_LEFT,
    VISION_TRACKS,
    StudentDisengaged,
    StudentIdentified,
    StudentLeft,
    TracksSeen,
    TrackView,
)
from lomas_core.events import EventBus
from lomas_core.schema import Config
from lomas_face import AttentionMonitor, IdentityMatcher, Tracker, estimate_pose
from lomas_face.types import Pose, Track
from lomas_vision import Frame, FrameBus

NO_SCALE = 1.0
MILLISECONDS = 1000.0
LEVEL = Pose(yaw=0.0, pitch=0.0, roll=0.0)
NO_SOURCE = ""


def downscale(image: np.ndarray, width: int) -> tuple[np.ndarray, float]:
    """Detect on a small copy, crop faces from the big one.

    The returned factor maps a box back to full-resolution pixels, which is
    what the embedder and the overlay both need.
    """
    height, full_width = image.shape[:2]
    if full_width <= width:
        return image, NO_SCALE

    import cv2  # only needed when a frame is actually bigger than the target

    factor = full_width / width
    small = cv2.resize(image, (width, int(height / factor)), interpolation=cv2.INTER_AREA)
    return small, factor


def source_for(cfg: Config) -> str:
    """Which camera feeds recognition. One robot has one; a CCTV install names
    the one pointed at the class and leaves the rest to the wall."""
    if cfg.vision.pipeline.source:
        return cfg.vision.pipeline.source
    enabled = [s.id for s in cfg.sources if s.enabled]
    return enabled[0] if enabled else NO_SOURCE


def vectors_by_student(rows: list[dict]) -> dict[str, list[np.ndarray]]:
    """Rebuild enrolled vectors from stored blobs. lomas_store keeps them as
    bytes so it never has to depend on numpy; this is where they come back."""
    enrolled: dict[str, list[np.ndarray]] = {}
    for row in rows:
        vector = np.frombuffer(row["vector"], dtype=row["dtype"])
        enrolled.setdefault(row["student_id"], []).append(vector)
    return enrolled


class VisionPipeline:
    """The one place frames meet faces.

    lomas_vision knows nothing about detection and lomas_face never opens a
    camera, so the join has to live somewhere the app owns. It runs on its own
    thread and always reads the newest frame, which means a slow cycle costs
    detail and never the lesson.
    """

    def __init__(
        self,
        cfg: Config,
        bus: EventBus,
        clock: Clock,
        frames: FrameBus,
        detector: Any,
        tracker: Tracker,
        matcher: IdentityMatcher,
        attention: AttentionMonitor,
    ) -> None:
        self.cfg = cfg
        self.bus = bus
        self.clock = clock
        self.frames = frames
        self.detector = detector
        self.tracker = tracker
        self.matcher = matcher
        self.attention = attention
        self.source_id = source_for(cfg)
        self.log = log.get("vision")

        self.cycles = 0
        self.skipped = 0
        self.errors = 0
        self.cycle_seconds = 0.0
        self.last_at = 0.0
        self._interval = NO_SCALE / cfg.face.detect_fps
        self._last_seq = 0
        self._identified: dict[int, tuple[str, float]] = {}
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    @property
    def runnable(self) -> bool:
        return bool(self.cfg.face.enabled and self.cfg.vision.pipeline.enabled and self.source_id)

    def load(self, enrolled: dict[str, list[np.ndarray]]) -> None:
        self.matcher.load(enrolled)
        self.log.debug("%s enrolled students loaded", len(enrolled))

    def start(self) -> None:
        if self._thread is not None:
            return
        if not self.runnable:
            self.log.info("vision is off; attendance falls back to the roster")
            return

        self.frames.start()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="vision", daemon=True)
        self._thread.start()
        self.log.info("vision running on %s at %s fps", self.source_id, self.cfg.face.detect_fps)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.cfg.vision.pipeline.join_timeout_seconds)
            self._thread = None

    def process(self, frame: Frame) -> list[Track]:
        """One detect cycle. Public because the thread is only a driver - a
        test can hand it frames and get the same behaviour with no threads."""
        began = time.perf_counter()
        small, factor = downscale(frame.image, self.cfg.face.downscale_width)
        detections = [d.scaled(factor) for d in self.detector.detect(small)]
        tracks = self.tracker.update(detections, frame.ts)

        for track in tracks:
            self._identify(track, frame, frame.ts)
            if track.box.landmarks is not None:
                track.pose = estimate_pose(track.box.landmarks, self.cfg.face.pose)

        self._departures(tracks, frame.ts)

        for signal in self.attention.update(tracks, frame.ts):
            self.bus.publish(
                STUDENT_DISENGAGED,
                StudentDisengaged(
                    track_id=signal.track_id,
                    student_id=signal.student_id,
                    score=signal.score,
                    drifting_for=signal.drifting_for,
                    at=signal.at,
                ),
            )

        if self.cfg.vision.pipeline.publish_tracks:
            self.bus.publish(VISION_TRACKS, self._view(frame, tracks))

        self.cycles += 1
        self.cycle_seconds += time.perf_counter() - began
        self.last_at = frame.ts
        return tracks

    def stats(self) -> dict[str, Any]:
        return {
            "cycles": self.cycles,
            "skipped": self.skipped,
            "errors": self.errors,
            "cycle_ms": round(self.cycle_seconds * MILLISECONDS / max(1, self.cycles), 1),
            "detect_fps_target": self.cfg.face.detect_fps,
            "running": self._thread is not None and self._thread.is_alive(),
            "tracks": len(self.tracker.active()),
            "identified": len(self._identified),
            **self.matcher.stats(),
        }

    def _run(self) -> None:
        """Nothing in here may reach the session.

        A camera that is unplugged mid-lesson, or a detector model that was
        never downloaded, has to end as a log line and a class that carries
        on - not as a dead thread nobody notices until the report is empty.
        """
        limits = self.cfg.vision.pipeline
        failures = 0

        while not self._stop.is_set():
            frame = self.frames.latest(self.source_id)
            if frame is None or frame.seq == self._last_seq:
                self.clock.sleep(limits.idle_sleep_seconds)
                continue

            # Frames captured between two cycles are meant to be missed:
            # capture runs at camera fps, detection deliberately slower.
            self.skipped += frame.seq - self._last_seq - 1
            self._last_seq = frame.seq

            try:
                self.process(frame)
                failures = 0
            except Exception as exc:
                failures += 1
                self.errors += 1
                self.log.error("detect cycle failed: %s", exc)
                if failures >= limits.max_consecutive_errors:
                    self.log.error("vision stopping; the class continues on the roster")
                    return

            self.clock.sleep(self._interval)

    def _identify(self, track: Track, frame: Frame, ts: float) -> None:
        """Recognition is skipped whole when the school has not consented.
        Tracks still exist, they simply carry no name."""
        if not self.cfg.privacy.recognition_enabled:
            return

        known = track.student_id
        # The full-resolution image, never the downscaled copy. Cropping from
        # the small one throws away the detail the embedder looks for.
        student_id = self.matcher.resolve(track, frame.image, ts)
        if not student_id or student_id == known:
            return

        self._identified[track.track_id] = (student_id, track.first_seen)
        self.bus.publish(
            STUDENT_IDENTIFIED,
            StudentIdentified(
                student_id=student_id,
                track_id=track.track_id,
                source_id=frame.source_id,
                zone=frame.zone,
                at=ts,
            ),
        )

    def _departures(self, tracks: list[Track], ts: float) -> None:
        live = {t.track_id for t in tracks}
        for track_id in [t for t in self._identified if t not in live]:
            student_id, first_seen = self._identified.pop(track_id)
            self.bus.publish(
                STUDENT_LEFT,
                StudentLeft(
                    student_id=student_id,
                    track_id=track_id,
                    seen_for=ts - first_seen,
                    at=ts,
                ),
            )

    def _view(self, frame: Frame, tracks: list[Track]) -> TracksSeen:
        return TracksSeen(
            source_id=frame.source_id,
            zone=frame.zone,
            at=frame.ts,
            width=frame.width,
            height=frame.height,
            tracks=tuple(
                TrackView(
                    track_id=t.track_id,
                    x=t.box.x,
                    y=t.box.y,
                    w=t.box.w,
                    h=t.box.h,
                    student_id=t.student_id,
                    attention=t.attention,
                    yaw=(t.pose or LEVEL).yaw,
                    pitch=(t.pose or LEVEL).pitch,
                    seen_for=t.age,
                )
                for t in tracks
            ),
        )
