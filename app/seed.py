from __future__ import annotations

from lomas_core import logging as log
from lomas_store import TenantScope

CONSENT_KIND = "face_recognition"
GRANTED_BY = "seed"

DEMO_STUDENTS = [
    ("Ananya Sharma", "01"),
    ("Rahul Deshmukh", "02"),
    ("Meera Patil", "03"),
    ("Kabir Shaikh", "04"),
    ("Zoya Ansari", "05"),
]


def demo_class(system) -> None:
    """Creates the org, school, class and a handful of students if they are
    not there yet, so a fresh checkout has something to teach.

    Idempotent - running it twice changes nothing.
    """
    cfg = system.cfg
    logger = log.get("seed")
    scope = TenantScope(
        org_id=cfg.active_org_id,
        school_id=cfg.tenancy.school_id,
        class_id=cfg.tenancy.class_id,
    )

    repos = system.repos
    repos["org"].ensure(scope, cfg.tenancy.org_id)

    if repos["school"].get(scope, scope.school_id) is None:
        repos["school"].create(scope, cfg.tenancy.school_id, cfg.content.language)

    if repos["class"].get(scope, scope.class_id) is None:
        repos["class"].create(scope, cfg.content.grade, "B", cfg.content.subject)

    existing = {s["roll_no"] for s in repos["student"].list_for_class(scope)}
    added = 0
    for name, roll in DEMO_STUDENTS:
        if roll in existing:
            continue
        student_id = repos["student"].create(scope, name, roll)
        repos["consent"].grant(scope, student_id, CONSENT_KIND, GRANTED_BY)
        added += 1

    if added:
        logger.info("seeded %d students into %s", added, scope.class_id)
