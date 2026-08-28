from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from lomas_store.store import STORES, Row

MEMORY_PATH = ":memory:"


class SqliteStore:
    """One connection guarded by a lock.

    Vision, the web server and the orchestrator all live in different threads
    and all touch this, so `check_same_thread` is off and every call goes
    through the lock instead.
    """

    def __init__(self, path: str, busy_timeout_ms: int) -> None:
        self.path = path
        if path != MEMORY_PATH:
            Path(path).parent.mkdir(parents=True, exist_ok=True)

        self._lock = threading.RLock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute(f"PRAGMA busy_timeout = {int(busy_timeout_ms)}")
        if path != MEMORY_PATH:
            self._conn.execute("PRAGMA journal_mode = WAL")

    def execute(self, sql: str, params: Sequence[Any] = ()) -> None:
        with self._lock:
            self._conn.execute(sql, tuple(params))
            self._conn.commit()

    def executemany(self, sql: str, rows: Iterable[Sequence[Any]]) -> None:
        with self._lock:
            self._conn.executemany(sql, [tuple(r) for r in rows])
            self._conn.commit()

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[Row]:
        with self._lock:
            cursor = self._conn.execute(sql, tuple(params))
            return [dict(row) for row in cursor.fetchall()]

    @contextmanager
    def transaction(self) -> Iterator[None]:
        with self._lock:
            try:
                yield
            except Exception:
                self._conn.rollback()
                raise
            self._conn.commit()

    def executescript(self, script: str) -> None:
        with self._lock:
            self._conn.executescript(script)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()


@STORES.register("sqlite")
class FileStore(SqliteStore):
    def __init__(self, path: str, busy_timeout_ms: int) -> None:
        super().__init__(path, busy_timeout_ms)


@STORES.register("memory")
class MemoryStore(SqliteStore):
    """Same engine, nothing on disk. Tests and dry runs use this."""

    def __init__(self, path: str = MEMORY_PATH, busy_timeout_ms: int = 0) -> None:
        super().__init__(MEMORY_PATH, busy_timeout_ms)
