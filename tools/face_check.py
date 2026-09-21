#!/usr/bin/env python3
"""How far away can this robot recognise a child, on this camera?

The numbers in the design are arithmetic: a 62 degree lens, 1280 pixels, a
face 14 cm wide, so about 2 m. Arithmetic is not a measurement, and the one
that matters - whether it says your name and not somebody else's - depends on
the lens, the light and the match threshold together.

    python tools/face_check.py --mode pi             # stand still, it measures
    python tools/face_check.py --mode pi --seconds 30

Stand at a metre, then two, then three, and watch the face width and the
distance to your own enrolment. What it prints at the end is the threshold
this camera in this room can actually hold.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "packages"))
sys.path.insert(0, str(ROOT))

from lomas_core.config import load  # noqa: E402
from lomas_core.errors import LomasError  # noqa: E402
from lomas_core.secrets import SECRETS_FILE, load_secrets  # noqa: E402
from lomas_face import DETECTORS, EMBEDDERS  # noqa: E402
from lomas_face.align import face_for  # noqa: E402
from lomas_face.embedder import distance  # noqa: E402

# A child's face, ear to ear, and the horizontal view of the Pi camera's
# standard lens. Both only to turn a face width in pixels into "about this
# far away", which is a sentence a person can act on.
FACE_CM = 14.0
LENS_DEGREES = 62.2
REPORT_EVERY = 1.0
SAFETY = 0.85  # a threshold is set inside the worst match seen, not on it


def build(cfg):
    detector = DETECTORS.create(cfg.face.detector, cfg.face)
    embedder = EMBEDDERS.create(cfg.face.embedder, cfg.face)
    return detector, embedder


def enrolled_vectors(cfg):
    """Everyone this robot knows, from its own database."""
    import numpy as np

    from lomas_store import STORES, EmbeddingRepo, StudentRepo, TenantScope, migrate

    store = STORES.create(cfg.storage.backend, cfg.storage.path, cfg.storage.busy_timeout_ms)
    migrate(store, time.time())
    scope = TenantScope(cfg.active_org_id, cfg.tenancy.school_id, cfg.tenancy.class_id)

    names = {row["id"]: row["name"] for row in StudentRepo(store).list_for_class(scope)}
    known: dict[str, list] = {}
    for row in EmbeddingRepo(store).all_for_class(scope):
        vector = np.frombuffer(row["vector"], dtype=row["dtype"])
        known.setdefault(names.get(row["student_id"], row["student_id"]), []).append(vector)
    store.close()
    return known


def metres_away(face_px: int, frame_px: int) -> float:
    """How far a face that wide is, roughly. One lens, one assumed head."""
    import math

    if face_px <= 0:
        return 0.0
    per_pixel = math.radians(LENS_DEGREES) / frame_px
    return (FACE_CM / 100.0 / 2) / math.tan(face_px * per_pixel / 2)


def nearest(vector, known) -> tuple[str, float]:
    best, score = "", 1.0
    for name, vectors in known.items():
        for other in vectors:
            gap = distance(vector, other)
            if gap < score:
                best, score = name, gap
    return best, score


def watch(cfg, seconds: float) -> int:
    from lomas_vision import CAMERA_SOURCES

    detector, embedder = build(cfg)
    known = enrolled_vectors(cfg)
    if not known:
        print("  nobody is enrolled on this robot yet. Enrol yourself first,")
        print("  then run this again - there is nothing to measure against.")
        return 1

    print(f"  {len(known)} enrolled: {', '.join(sorted(known))}")
    print(f"  matching at {cfg.face.match_threshold} (lower is stricter), "
          f"align {'on' if cfg.face.align else 'off'}\n")

    spec = next(s for s in cfg.sources if s.enabled)
    camera = CAMERA_SOURCES.create(spec.kind, spec)
    camera.open()

    print("  face px   about      who it thinks      distance")
    print("  -------   -------    --------------     --------")
    seen: list[tuple[int, str, float]] = []
    until = time.monotonic() + seconds
    last = 0.0
    try:
        while time.monotonic() < until:
            frame = camera.read()
            if frame is None:
                continue
            faces = detector.detect(frame.image)
            if not faces or time.monotonic() - last < REPORT_EVERY:
                continue
            last = time.monotonic()

            face = max(faces, key=lambda d: d.w)
            crop = face_for(embedder, frame.image, face, cfg.face.crop_margin, cfg.face.align)
            if not crop.size:
                continue

            who, gap = nearest(embedder.embed(crop), known)
            away = metres_away(face.w, frame.image.shape[1])
            matched = gap <= cfg.face.match_threshold
            print(f"  {face.w:>5} px   {away:>4.1f} m    {who:<18} {gap:.3f}"
                  f"   {'' if matched else '(too far off to be a match)'}")
            seen.append((face.w, who, gap))
    except KeyboardInterrupt:
        pass
    finally:
        camera.close()

    return report(seen, cfg)


def report(seen, cfg) -> int:
    if not seen:
        print("\n  no face was ever detected. Check the camera and the light.")
        return 1

    matched = [(px, gap) for px, _who, gap in seen if gap <= cfg.face.match_threshold]
    print(f"\n  {len(seen)} looks, {len(matched)} of them matched.")

    if matched:
        widest = max(px for px, _ in matched)
        narrowest = min(px for px, _ in matched)
        worst = max(gap for _, gap in matched)
        print(f"  recognised from {narrowest} px to {widest} px wide")
        print(f"    which is about {metres_away(widest, 1280):.1f} m "
              f"to {metres_away(narrowest, 1280):.1f} m away")
        print(f"  the worst match that still worked: {worst:.3f}")
        print(f"\n  A threshold of {max(worst / SAFETY, worst + 0.02):.2f} would hold this "
              "room with a little room to spare.")
        print("  Lower is stricter: a robot that says nothing beats one that")
        print("  says the wrong child's name in front of the class.")

    missed = [px for px, _who, gap in seen if gap > cfg.face.match_threshold]
    if missed:
        print(f"\n  {len(missed)} looks did not match, the widest at {max(missed)} px.")
        print("  If that was you standing close, the enrolment is the problem:")
        print("  enrol again in this light, looking left, straight and right.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="measure recognition on this camera")
    parser.add_argument("--mode", default="pi")
    parser.add_argument("--config-dir", default=str(ROOT / "config"))
    parser.add_argument("--seconds", type=float, default=60.0)
    args = parser.parse_args()

    load_secrets(Path(args.config_dir) / SECRETS_FILE)
    cfg = load(args.config_dir, args.mode)

    print("=== stand in front of the camera and move back a step at a time ===")
    try:
        return watch(cfg, args.seconds)
    except LomasError as exc:
        print(f"  {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
