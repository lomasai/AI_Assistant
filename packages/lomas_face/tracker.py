from __future__ import annotations

from itertools import count

from lomas_core.schema import FaceConfig
from lomas_face.types import Detection, Track

NO_OVERLAP = 0.0


def iou(left: Detection, right: Detection) -> float:
    ax1, ay1, ax2, ay2 = left.box
    bx1, by1, bx2, by2 = right.box

    inter_w = min(ax2, bx2) - max(ax1, bx1)
    inter_h = min(ay2, by2) - max(ay1, by1)
    if inter_w <= 0 or inter_h <= 0:
        return NO_OVERLAP

    overlap = inter_w * inter_h
    union = left.area + right.area - overlap
    return overlap / union if union else NO_OVERLAP


class Tracker:
    """Associates detections to existing tracks by overlap.

    Identity is expensive and does not change between frames, so the whole
    point of this class is to give a face a stable id that a recognised name
    can be attached to once.
    """

    def __init__(self, cfg: FaceConfig) -> None:
        self.cfg = cfg
        self._tracks: dict[int, Track] = {}
        self._ids = count(1)

    def update(self, detections: list[Detection], ts: float) -> list[Track]:
        usable = [
            d
            for d in detections
            if d.confidence >= self.cfg.min_confidence and d.w >= self.cfg.min_face_px
        ]

        matched_tracks, matched_dets = self._associate(usable)

        for track_id, det_index in matched_tracks.items():
            track = self._tracks[track_id]
            track.box = usable[det_index]
            track.last_seen = ts
            track.hits += 1
            track.misses = 0
            track.history.append(track.box.centre)
            if track.hits >= self.cfg.track_birth_hits:
                track.confirmed = True

        for track_id, track in list(self._tracks.items()):
            if track_id in matched_tracks:
                continue
            track.misses += 1
            if ts - track.last_seen >= self.cfg.track_death_seconds:
                del self._tracks[track_id]

        for index, detection in enumerate(usable):
            if index in matched_dets or len(self._tracks) >= self.cfg.max_tracks:
                continue
            track_id = next(self._ids)
            self._tracks[track_id] = Track(
                track_id=track_id,
                box=detection,
                first_seen=ts,
                last_seen=ts,
                history=[detection.centre],
                confirmed=self.cfg.track_birth_hits <= 1,
            )

        return self.active()

    def active(self) -> list[Track]:
        return [t for t in self._tracks.values() if t.confirmed]

    def all_tracks(self) -> list[Track]:
        return list(self._tracks.values())

    def reset(self) -> None:
        self._tracks.clear()

    def _associate(self, detections: list[Detection]) -> tuple[dict[int, int], set[int]]:
        """Greedy best-overlap first.

        Taking the strongest pair first is what stops two faces passing each
        other from trading ids: each keeps the box it overlaps most.
        """
        candidates = [
            (iou(track.box, detection), track_id, index)
            for track_id, track in self._tracks.items()
            for index, detection in enumerate(detections)
        ]
        candidates.sort(reverse=True)

        matched_tracks: dict[int, int] = {}
        matched_dets: set[int] = set()
        for score, track_id, index in candidates:
            if score < self.cfg.track_iou_threshold:
                break
            if track_id in matched_tracks or index in matched_dets:
                continue
            matched_tracks[track_id] = index
            matched_dets.add(index)

        return matched_tracks, matched_dets
