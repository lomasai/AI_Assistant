from __future__ import annotations

from pathlib import Path

from lomas_store.store import Store

SCHEMA_FILE = Path(__file__).with_name("schema.sql")

VERSION_TABLE = """
CREATE TABLE IF NOT EXISTS schema_version (
    version    INTEGER PRIMARY KEY,
    applied_at REAL NOT NULL
)
"""


def _steps() -> list[tuple[int, str]]:
    return [(1, SCHEMA_FILE.read_text(encoding="utf-8"))]


def current_version(store: Store) -> int:
    store.execute(VERSION_TABLE)
    rows = store.query("SELECT MAX(version) AS v FROM schema_version")
    return rows[0]["v"] or 0


def migrate(store: Store, now: float) -> int:
    """Forward only. Running it twice applies nothing the second time."""
    applied = current_version(store)
    for version, script in _steps():
        if version <= applied:
            continue
        store.executescript(script)
        store.execute(
            "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
            (version, now),
        )
        applied = version
    return applied
