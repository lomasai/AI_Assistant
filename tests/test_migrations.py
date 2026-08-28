from __future__ import annotations

import pytest

from lomas_store import STORES, migrate
from lomas_store.migrations import current_version

SEED_TIME = 1_700_000_000.0

EXPECTED_TABLES = {
    "orgs", "schools", "classes", "students", "consent",
    "embeddings", "sessions", "session_students", "answers", "events",
}


@pytest.fixture
def store():
    backing = STORES.create("memory")
    yield backing
    backing.close()


def table_names(store) -> set[str]:
    rows = store.query("SELECT name FROM sqlite_master WHERE type = 'table'")
    return {row["name"] for row in rows}


def test_migrate_creates_every_table(store):
    migrate(store, SEED_TIME)
    assert EXPECTED_TABLES <= table_names(store)


def test_migrate_is_idempotent(store):
    first = migrate(store, SEED_TIME)
    second = migrate(store, SEED_TIME)
    assert first == second == current_version(store)

    applied = store.query("SELECT COUNT(*) AS n FROM schema_version")
    assert applied[0]["n"] == first, "a second run must apply nothing"


def test_no_table_can_hold_an_image(store):
    """The schema has nowhere to put a face image, and that is deliberate."""
    migrate(store, SEED_TIME)
    banned = ("image", "photo", "frame", "picture", "thumbnail", "video")

    for table in EXPECTED_TABLES:
        columns = {row["name"].lower() for row in store.query(f"PRAGMA table_info({table})")}
        offending = {c for c in columns if any(word in c for word in banned)}
        assert not offending, f"{table} gained image-shaped column(s) {offending}"


def test_file_backend_is_registered():
    assert "sqlite" in STORES
    assert "memory" in STORES
