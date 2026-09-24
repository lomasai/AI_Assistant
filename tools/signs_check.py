#!/usr/bin/env python3
"""What can this robot actually see, and what does it cost?

Two questions, both of which decide whether signing is worth having:

  * how far away a card still reads, on this camera in this room's light
  * what a read costs on this machine - because the hand model is the one
    expensive thing here, and "too heavy" is a measurement, not a feeling

    python tools/signs_check.py --mode pi                  # cards only
    python tools/signs_check.py --mode pi --hands          # and the hand model
    python tools/signs_check.py --mode pi --seconds 60

Hold a card up and walk backwards. It prints, once a second, which markers
it can see, how wide they are in the picture, and roughly how far away that
is. Stop where it stops seeing you: that is this classroom's range.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "packages"))
sys.path.insert(0, str(ROOT))

from lomas_core.clock import RealClock  # noqa: E402
from lomas_core.config import load  # noqa: E402
from lomas_core.secrets import SECRETS_FILE, load_secrets  # noqa: E402
from lomas_face import DETECTORS  # noqa: E402
from lomas_signs import CARD_READERS, HAND_READERS  # noqa: E402
from lomas_signs.types import Box  # noqa: E402
from lomas_vision import FrameBus, build_sources, downscale  # noqa: E402

# Only to turn a marker width in pixels into "about this far", which is a
# sentence somebody can act on. The card's printed size and the camera's
# horizontal view.
CARD_CM = 10.0
LENS_DEGREES = 62.2
REPORT_EVERY = 1.0
MILLISECONDS = 1000.0
HALF = 2.0
CENTIMETRES = 100.0
SHOT = "signs_seen.png"
FACE_COLOUR = (120, 200, 120)
SIGN_COLOUR = (80, 120, 255)
CARD_COLOUR = (255, 200, 80)
LINE = 2
TEXT_SIZE = 0.6
TEXT_UP = 8


def distance_m(width_px: float, frame_px: int) -> float:
    """How far away a card of the printed size would have to be to appear
    this wide. Arithmetic, not a measurement - but it makes the numbers
    readable while you walk backwards."""
    import math

    if width_px <= 0:
        return 0.0
    per_radian = frame_px / (HALF * math.tan(math.radians(LENS_DEGREES) / HALF))
    return (CARD_CM / CENTIMETRES) * per_radian / width_px


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", default="pi")
    ap.add_argument("--config-dir", default=str(ROOT / "config"))
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--hands", action="store_true", help="measure hands too")
    ap.add_argument("--save", action="store_true",
                    help="write a picture of what it matched, into data/logs")
    args = ap.parse_args()

    load_secrets(Path(args.config_dir) / SECRETS_FILE)
    overrides = ["signs.enabled=true"]
    cfg = load(args.config_dir, args.mode, overrides, use_env=True)
    if args.hands and cfg.signs.hands.reader == "none":
        # Asked for hands on a profile that has them off: measure the one
        # that needs nothing installed.
        cfg = load(args.config_dir, args.mode,
                   [*overrides, "signs.hands.reader=raised_hand"], use_env=True)

    cards = CARD_READERS.create(cfg.signs.cards.reader, cfg.signs.cards)
    hands = HAND_READERS.create(cfg.signs.hands.reader, cfg.signs.hands)
    # A raised hand is found beside a face, so this has to find the faces -
    # and the total it prints is then what a cycle really costs.
    detector = DETECTORS.create(cfg.face.detector, cfg.face) if args.hands else None
    print(f"cards: {cards.describe()}")
    print(f"hands: {hands.describe()}")
    if not cards.available and not hands.available:
        print("\nNothing can read a sign. Cards need opencv; hands need "
              "`pip install mediapipe` and the .task model.")
        return 1

    frames = FrameBus(
        build_sources(cfg.sources),
        buffer_size=cfg.vision.buffer_size,
        clock=RealClock(),
        read_timeout_ms=cfg.vision.read_timeout_ms,
    )
    frames.start()
    source = cfg.signs.source or cfg.sources[0].id

    shown = Path(cfg.runtime.log_dir) / SHOT
    print(f"\nWatching {source} for {args.seconds:.0f}s. Hold a card up and walk back.\n")
    if args.save:
        print(f"...and writing what it matched to {shown}\n")
    print(f"{'cards':>6} {'widest':>7} {'~far':>6} {'card ms':>8} {'hand ms':>8}  seen")

    card_ms: list[float] = []
    hand_ms: list[float] = []
    face_ms: list[float] = []
    widest = 0
    furthest = 0.0
    ends = time.monotonic() + args.seconds
    next_report = time.monotonic()
    last_seq = 0

    try:
        while time.monotonic() < ends:
            frame = frames.latest(source)
            if frame is None or frame.seq == last_seq:
                time.sleep(0.01)
                continue
            last_seq = frame.seq

            # The same small copy the robot reads, or the cost measured
            # here is not the cost it pays.
            small, factor = downscale(frame.image, cfg.signs.downscale_width)

            began = time.perf_counter()
            seen = cards.read(small, frame.ts)
            card_ms.append((time.perf_counter() - began) * MILLISECONDS)

            signs = []
            if hands.available:
                # Timed apart: the robot does not find faces for this. It
                # reads the ones the lesson's own detector already found, so
                # only the second number is what signing adds.
                began = time.perf_counter()
                faces = _faces(detector, small)
                face_ms.append((time.perf_counter() - began) * MILLISECONDS)

                began = time.perf_counter()
                signs = hands.read(small, frame.ts, faces)
                hand_ms.append((time.perf_counter() - began) * MILLISECONDS)

            if time.monotonic() < next_report:
                continue
            next_report = time.monotonic() + REPORT_EVERY

            wide = int(max((card.box.w for card in seen), default=0) * factor)
            far = distance_m(wide, frame.image.shape[1]) if wide else 0.0
            widest = max(widest, wide)
            furthest = max(furthest, far)
            names = ", ".join(f"#{c.marker_id}/{c.turn}" for c in seen) or "-"
            if signs:
                names += " | " + ", ".join(f"{s.name} {s.score:.2f}" for s in signs)
            if args.save:
                _picture(small, faces, signs, seen, shown)
            print(f"{len(seen):>6} {wide:>7} {far:>5.1f}m {_mean(card_ms):>7.1f} "
                  f"{_mean(hand_ms):>7.1f}  {names}")
    except KeyboardInterrupt:
        pass
    finally:
        frames.stop()
        hands.close()

    print("\n--- what this camera in this room can do ---")
    print(f"a card read at up to about {furthest:.1f} m (widest {widest} px)")
    print(f"reading cards costs {_mean(card_ms):.1f} ms a frame")
    if hand_ms:
        cost = _mean(hand_ms) * cfg.signs.fps / MILLISECONDS
        print(f"reading hands costs {_mean(hand_ms):.1f} ms a read, which at "
              f"{cfg.signs.fps} reads a second is {cost * 100:.0f}% of one core")
        print(f"(finding the faces to look beside took another {_mean(face_ms):.1f} ms "
              "here, which the robot does not pay twice - the lesson's own "
              "detector has already found them)")
        print("Too dear? Raise signs.hands.every, or signs.hands.reader: none.")
    else:
        print("hands were not measured (--hands, and mediapipe installed)")
    return 0


def _picture(image, faces, signs, cards, where: Path) -> None:
    """What the robot is looking at, with what it found drawn on it.

    Written because guessing twice is a habit: the reader said `raised_hand`
    on every frame of a room with nobody's hand up, and no amount of reading
    the code says which beige thing it was.
    """
    import cv2

    shot = image.copy()
    for face in faces:
        cv2.rectangle(shot, (face.x, face.y), (face.x + face.w, face.y + face.h),
                      FACE_COLOUR, LINE)
    for sign in signs:
        box = sign.box
        cv2.rectangle(shot, (box.x, box.y), (box.x + box.w, box.y + box.h),
                      SIGN_COLOUR, LINE)
        cv2.putText(shot, sign.name, (box.x, max(box.y - TEXT_UP, TEXT_UP)),
                    cv2.FONT_HERSHEY_SIMPLEX, TEXT_SIZE, SIGN_COLOUR, LINE)
    for card in cards:
        box = card.box
        cv2.rectangle(shot, (box.x, box.y), (box.x + box.w, box.y + box.h),
                      CARD_COLOUR, LINE)

    where.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(where), shot)


def _faces(detector, image) -> tuple[Box, ...]:
    if detector is None:
        return ()
    return tuple(Box(x=found.x, y=found.y, w=found.w, h=found.h)
                 for found in detector.detect(image))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
