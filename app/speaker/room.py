from __future__ import annotations

import threading

from lomas_core.clock import Clock
from lomas_core.contracts import STUDENT_IDENTIFIED, VISION_TRACKS, StudentIdentified, TracksSeen
from lomas_core.events import EventBus
from lomas_core.schema import SpeakerConfig


class Room:
    """Who the camera can see right now.

    Attendance says who came to the lesson; this says who is in front of the
    robot this second, which is a different question and the one that decides
    whether a name needs saying at all.
    """

    def __init__(self, cfg: SpeakerConfig, bus: EventBus, clock: Clock) -> None:
        self.cfg = cfg
        self.clock = clock
        self._seen: dict[str, float] = {}
        self._moving: dict[str, float] = {}
        self._lock = threading.Lock()

        bus.subscribe(VISION_TRACKS, self._on_tracks)
        # Recognition fires once per track; without this a child who has been
        # identified but whose track carries no id reads as absent.
        bus.subscribe(STUDENT_IDENTIFIED, self._on_identified)

    def visible(self) -> list[str]:
        cutoff = self.clock.now() - self.cfg.visible_seconds
        with self._lock:
            return sorted(who for who, at in self._seen.items() if at >= cutoff)

    def mouths(self) -> dict[str, float]:
        """How much each visible child's mouth is moving. Only faces the
        camera can still see: a score left behind by somebody who walked out
        would answer for them."""
        here = set(self.visible())
        with self._lock:
            return {who: score for who, score in self._moving.items() if who in here}

    def _on_tracks(self, _event: str, seen: TracksSeen) -> None:
        # Tolerant of what arrives: the debug surface and the tests publish
        # their own shapes on this event, and none of them may stop a class.
        now = self.clock.now()
        with self._lock:
            for track in getattr(seen, "tracks", ()):
                if getattr(track, "student_id", ""):
                    self._seen[track.student_id] = now
                    self._moving[track.student_id] = getattr(track, "mouth", 0.0)

    def _on_identified(self, _event: str, who: StudentIdentified) -> None:
        student_id = getattr(who, "student_id", "")
        if not student_id:
            return
        with self._lock:
            self._seen[student_id] = self.clock.now()
