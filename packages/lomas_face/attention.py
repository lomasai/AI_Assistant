from __future__ import annotations

import math

from lomas_core.schema import AttentionConfig
from lomas_face.types import Disengagement, Track

ENGAGED = 1.0
AWAY = 0.0
NEVER = 0.0
FULL_ALPHA = 1.0


class AttentionMonitor:
    """Scores whether each tracked face is roughly pointed at the robot, and
    says when someone has been elsewhere long enough to be worth a question.

    It reports drift. It does not decide what to do about it - that belongs
    to the engagement agent, which is deliberately the only thing that can
    turn this into speech.
    """

    def __init__(self, cfg: AttentionConfig) -> None:
        self.cfg = cfg
        self._scores: dict[int, float] = {}
        self._seen_at: dict[int, float] = {}
        self._drifting_since: dict[int, float] = {}
        self._last_nudge: dict[str, float] = {}
        self._nudges = 0

    def update(self, tracks: list[Track], ts: float) -> list[Disengagement]:
        if not self.cfg.enabled:
            return []

        signals: list[Disengagement] = []
        live = {t.track_id for t in tracks}
        for stale in set(self._scores) - live:
            self._scores.pop(stale, None)
            self._seen_at.pop(stale, None)
            self._drifting_since.pop(stale, None)

        for track in tracks:
            score = self._score(track, ts)
            track.attention = score
            if score >= self.cfg.threshold:
                self._drifting_since.pop(track.track_id, None)
                continue

            since = self._drifting_since.setdefault(track.track_id, ts)
            drifting_for = ts - since
            if drifting_for < self.cfg.min_duration_seconds:
                continue
            if not self._may_nudge(track, ts):
                continue

            self._record_nudge(track, ts)
            signals.append(
                Disengagement(
                    track_id=track.track_id,
                    student_id=track.student_id,
                    score=score,
                    drifting_for=drifting_for,
                    at=ts,
                )
            )

        return signals

    def score_of(self, track_id: int) -> float:
        return self._scores.get(track_id, ENGAGED)

    def reset(self) -> None:
        """Called when a session ends. Nudge budgets are per session."""
        self._scores.clear()
        self._seen_at.clear()
        self._drifting_since.clear()
        self._last_nudge.clear()
        self._nudges = 0

    def _score(self, track: Track, ts: float) -> float:
        previous = self._scores.get(track.track_id, ENGAGED)
        last_seen = self._seen_at.get(track.track_id)
        self._seen_at[track.track_id] = ts

        # No landmarks means no opinion. A child who turns fully away stops
        # being detected at all, and that is a departure, not inattention.
        if track.pose is None:
            self._scores[track.track_id] = previous
            return previous

        facing = (
            abs(track.pose.yaw) <= self.cfg.cone_yaw_degrees
            and abs(track.pose.pitch) <= self.cfg.cone_pitch_degrees
        )
        target = ENGAGED if facing else AWAY

        # Time-constant EMA, so a variable frame rate does not change how
        # quickly the score moves.
        elapsed = ts - last_seen if last_seen is not None else NEVER
        alpha = FULL_ALPHA if elapsed <= NEVER else 1.0 - math.exp(-elapsed / self.cfg.window_seconds)
        updated = previous + alpha * (target - previous)
        self._scores[track.track_id] = updated
        return updated

    def _key(self, track: Track) -> str:
        return track.student_id or f"track:{track.track_id}"

    def _may_nudge(self, track: Track, ts: float) -> bool:
        if self._nudges >= self.cfg.max_nudges_per_session:
            return False
        last = self._last_nudge.get(self._key(track))
        return last is None or (ts - last) >= self.cfg.cooldown_seconds

    def _record_nudge(self, track: Track, ts: float) -> None:
        self._last_nudge[self._key(track)] = ts
        self._nudges += 1
        self._drifting_since[track.track_id] = ts

