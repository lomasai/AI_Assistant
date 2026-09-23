from __future__ import annotations

import threading
import time
from typing import Any

from lomas_core import logging as log
from lomas_core.clock import Clock
from lomas_core.contracts import (
    CARD_SEEN,
    HAND_UP,
    SIGN_SEEN,
    VISION_TRACKS,
    CardSeen,
    HandUp,
    SignSeen,
)
from lomas_core.events import EventBus
from lomas_core.schema import Config
from lomas_signs import CARD_READERS, HAND_READERS, Card, Sign
from lomas_store import TenantScope

from app.pipeline import downscale, source_for

ASK = "ask"
YES = "yes"
NO = "no"
FAR = float("inf")
HALF = 2
CARD = "card"
HAND = "hand"
NOBODY = ""
ONE = 1


class SignWatch:
    """Watches the room for a raised hand or a card held up.

    Its own thread and its own cadence, because this is not face detection.
    A sign meant as an interrupt is held for a second or two, so a few reads
    a second catch it as surely as eight would - and on a Pi that is already
    running a detector, the frames you do not read are the whole budget.

    It says what was seen and never what to do about it. Which sign means
    "I want to ask" is config, and what happens next belongs to the step
    that is teaching.
    """

    def __init__(self, cfg: Config, bus: EventBus, clock: Clock, frames: Any,
                 repos: dict[str, Any], scope_of=None) -> None:
        self.cfg = cfg
        self.bus = bus
        self.clock = clock
        self.frames = frames
        self.repos = repos
        self.scope_of = scope_of or (lambda: TenantScope(org_id=cfg.active_org_id))
        self.log = log.get("signs")

        self.cards = CARD_READERS.create(cfg.signs.cards.reader, cfg.signs.cards)
        self.hands = HAND_READERS.create(cfg.signs.hands.reader, cfg.signs.hands)

        self.reads = 0
        self.errors = 0
        self.read_seconds = 0.0
        self.seen_cards = 0
        self.seen_signs = 0

        self._tracks: list[Any] = []
        self._held: dict[str, int] = {}
        self._recent: dict[str, tuple[str, float]] = {}
        self._cards_now: dict[int, tuple[Card, float]] = {}
        self._source = cfg.signs.source or source_for(cfg)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._last_seq = 0

        bus.subscribe(VISION_TRACKS, self._on_tracks)

    @property
    def runnable(self) -> bool:
        return bool(self.cfg.signs.enabled and self._source
                    and (self.cards.available or self.hands.available))

    def describe(self) -> str:
        return f"{self.cards.describe()}, {self.hands.describe()}"

    # --- the thread --------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None or not self.runnable:
            if self.cfg.signs.enabled and not self.runnable:
                self.log.info("signs are on but nothing can read one: %s", self.describe())
            return
        self.frames.start()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="signs", daemon=True)
        self._thread.start()
        self.log.info("watching for signs on %s at %s reads a second (%s)",
                      self._source, self.cfg.signs.fps, self.describe())

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.cfg.signs.join_timeout_seconds)
            self._thread = None
        self.hands.close()

    def _run(self) -> None:
        gap = ONE / self.cfg.signs.fps
        while not self._stop.is_set():
            began = time.perf_counter()
            try:
                frame = self.frames.latest(self._source)
                if frame is not None and frame.seq != self._last_seq:
                    self._last_seq = frame.seq
                    self.read(frame)
            except Exception as exc:
                # A camera unplugged mid-class is a class without signs, not
                # a class that stops.
                self.errors += 1
                self.log.debug("sign read failed: %s", exc)
            waiting = gap - (time.perf_counter() - began)
            if waiting > 0:
                self._stop.wait(waiting)

    # --- one read ----------------------------------------------------------

    def read(self, frame) -> None:
        """One look at the room. Public because the thread is only a driver:
        a test hands it frames and gets the same behaviour with no threads."""
        began = time.perf_counter()
        small, factor = downscale(frame.image, self.cfg.signs.downscale_width)

        for card in self.cards.read(small, frame.ts):
            self._on_card(card, factor, frame.ts)

        if self.hands.available and self.reads % self.cfg.signs.hands.every == 0:
            for sign in self.hands.read(small, frame.ts):
                self._on_sign(sign, factor, frame.ts)

        self.reads += 1
        self.read_seconds += time.perf_counter() - began

    def _on_card(self, card: Card, factor: float, at: float) -> None:
        card = Card(marker_id=card.marker_id, box=card.box.scaled(factor), turn=card.turn, at=at)
        self._cards_now[card.marker_id] = (card, at)
        if not self._steady(f"{CARD}:{card.marker_id}:{card.turn}",
                            self.cfg.signs.cards.hold_reads):
            return

        student_id, name = self._owner(card.marker_id)
        answers = self.cfg.signs.cards.answers
        self.seen_cards += 1
        self.bus.publish(CARD_SEEN, CardSeen(
            marker_id=card.marker_id, turn=card.turn,
            answer=answers[card.turn] if card.turn < len(answers) else "",
            student_id=student_id, student_name=name, source_id=self._source, at=at))

        if card.upright and self.cfg.signs.cards.ask_when_upright:
            self._wants_to_ask(student_id, name, CARD, at)

    def _on_sign(self, sign: Sign, factor: float, at: float) -> None:
        means = self.cfg.signs.hands.actions.get(sign.name, NOBODY)
        if not means:
            return
        if not self._steady(f"{HAND}:{sign.name}", self.cfg.signs.hands.hold_reads):
            return

        student_id, name = self._whose(sign.box.scaled(factor))
        self.seen_signs += 1
        self._recent[means] = (sign.name, at)
        self.bus.publish(SIGN_SEEN, SignSeen(
            name=sign.name, means=means, student_id=student_id, student_name=name,
            score=sign.score, source_id=self._source, at=at))

        if means == ASK:
            self._wants_to_ask(student_id, name, HAND, at)

    def _wants_to_ask(self, student_id: str, name: str, by: str, at: float) -> None:
        self.bus.publish(HAND_UP, HandUp(student_id=student_id, student_name=name, by=by, at=at))

    # --- what a sign is worth ---------------------------------------------

    def _steady(self, key: str, needed: int) -> bool:
        """Held, rather than flashed.

        A hand halfway through a stretch and a card on its way into a bag
        both appear for one read. Waiting for a few in a row is also what
        separates a sign from a wave, which is motion by definition.
        """
        self._held[key] = self._held.get(key, 0) + ONE
        if self._held[key] < needed:
            return False
        self._held[key] = 0
        return True

    def agreement(self, within_seconds: float = 0.0) -> str:
        """Whether somebody has just nodded or shaken their head at the
        robot, for a question that was asked out loud."""
        window = within_seconds or self.cfg.signs.remember_seconds
        now = self.clock.now()
        for means in (YES, NO):
            seen = self._recent.get(means)
            if seen is not None and now - seen[1] <= window:
                return means
        return NOBODY

    def visible_cards(self, within_seconds: float = 0.0) -> list[Card]:
        window = within_seconds or self.cfg.signs.remember_seconds
        now = self.clock.now()
        return [card for card, at in self._cards_now.values() if now - at <= window]

    def card_owners(self) -> dict[int, tuple[str, str]]:
        return {card.marker_id: self._owner(card.marker_id) for card in self.visible_cards()}

    # --- who ---------------------------------------------------------------

    def _owner(self, marker_id: int) -> tuple[str, str]:
        row = self.repos["card"].owner(self.scope_of(), marker_id)
        return (row["student_id"], row["name"]) if row else (NOBODY, NOBODY)

    def _whose(self, box) -> tuple[str, str]:
        """The named face nearest this hand.

        A hand on its own says nothing about who is asking, which is the
        whole reason cards came first - but when a face is recognised right
        beside it, that is whose hand it is.
        """
        near = self.cfg.signs.hands.near_face
        hand_x, hand_y = box.centre
        best: tuple[float, Any] = (FAR, None)
        for track in self._tracks:
            if not getattr(track, "student_id", None):
                continue
            gap = abs((track.x + track.w / HALF) - hand_x) + abs((track.y + track.h / HALF) - hand_y)
            # A hand belongs to a face within a few face-widths of it. Any
            # further and it is somebody else's, or nobody's.
            if gap < best[0] and gap <= track.w / near:
                best = (gap, track)

        found = best[1]
        if found is None:
            return NOBODY, NOBODY
        return found.student_id or NOBODY, self._name_of(found.student_id)

    def _name_of(self, student_id: str) -> str:
        if not student_id:
            return NOBODY
        row = self.repos["student"].get(self.scope_of(), student_id)
        return row["name"] if row else NOBODY

    def _on_tracks(self, _event: str, seen) -> None:
        self._tracks = list(getattr(seen, "tracks", ()) or ())

    def stats(self) -> dict[str, Any]:
        average = (self.read_seconds / self.reads * 1000.0) if self.reads else 0.0
        return {
            "reads": self.reads,
            "errors": self.errors,
            "cards": self.seen_cards,
            "signs": self.seen_signs,
            "read_ms": round(average, 1),
            "reader": self.describe(),
        }
