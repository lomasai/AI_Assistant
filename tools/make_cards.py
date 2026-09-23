#!/usr/bin/env python3
"""Print a card for every child in the class.

Nothing is bought and nothing is downloaded: the markers are generated here
and printed on whatever printer the school has. Each card carries one child's
marker, their name, and a letter on each edge - so holding it with B at the
top is answering B, and a whole class answers a question in one frame.

    python tools/make_cards.py --mode pi               # a sheet for the class
    python tools/make_cards.py --mode pi --issue       # ...and record who has which
    python tools/make_cards.py --mode pi --spare 6     # blanks for new children

Print on matte paper, not glossy: gloss throws a highlight into the camera
and the detector loses the square. Cut on the white, never into it - the
quiet border is what makes a marker readable.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "packages"))
sys.path.insert(0, str(ROOT))

from lomas_core.config import load  # noqa: E402
from lomas_core.errors import LomasError  # noqa: E402
from lomas_signs.sheet import capacity, card, sheet  # noqa: E402
from lomas_store import STORES, CardRepo, StudentRepo, TenantScope, migrate  # noqa: E402

PAGE = "cards.png"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", default="debug")
    ap.add_argument("--issue", action="store_true",
                    help="record in the database which child has which card")
    ap.add_argument("--spare", type=int, default=0, help="extra blank cards to print")
    ap.add_argument("--out", default="", help="where to write the sheet")
    args = ap.parse_args()

    cfg = load(str(ROOT / "config"), args.mode, [], use_env=True)
    settings = cfg.signs.cards

    store = STORES.create(cfg.storage.backend, cfg.storage.path, cfg.storage.busy_timeout_ms)
    migrate(store, 0.0)
    scope = TenantScope(org_id=cfg.active_org_id, school_id=cfg.tenancy.school_id,
                        class_id=cfg.tenancy.class_id)
    students = StudentRepo(store).list_for_class(scope)
    cards = CardRepo(store)

    if not students and not args.spare:
        raise LomasError("no children in this class, and no --spare asked for")

    room = capacity(settings.dictionary)
    if len(students) + args.spare > room:
        raise LomasError(
            f"{settings.dictionary} holds {room} different cards and "
            f"{len(students) + args.spare} were asked for. Use a bigger family, "
            "such as signs.cards.dictionary: DICT_4X4_250."
        )

    faces = []
    for index, student in enumerate(students):
        # The roll number order, so the sheet comes off the printer in the
        # order the register is read.
        # The roll number too: it is what a teacher reads off a register
        # when handing the cards out.
        faces.append(card(settings.dictionary, index,
                          f"{student['roll_no']} {student['name']}", settings.print_px,
                          settings.answers))
        if args.issue:
            cards.issue(scope, student["id"], index)

    # Spares carry on where the class stopped, so a child who joins in
    # March gets the next number and nobody has two of anything.
    for spare in range(args.spare):
        faces.append(card(settings.dictionary, len(students) + spare, "", settings.print_px,
                          settings.answers))

    page = sheet(faces, settings.print_across, settings.print_gap_px)
    out = Path(args.out) if args.out else Path(settings.sheet_dir) / PAGE
    out.parent.mkdir(parents=True, exist_ok=True)

    import cv2

    cv2.imwrite(str(out), page)

    print(f"{len(faces)} cards -> {out}")
    for index, student in enumerate(students):
        print(f"  {index:>3}  {student['roll_no']:>4}  {student['name']}")
    if args.spare:
        print(f"  {len(students)}+  spare, for children who join later")
    if not args.issue:
        print("\nNobody has been given one yet. Run again with --issue once the "
              "cards are cut and handed out.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
