from __future__ import annotations

import json

from lomas_store.repos.base import Repository
from lomas_store.scope import TenantScope
from lomas_store.store import Row


class EventRepo(Repository):
    """Append-only session log. Every report reads from here, so there is no
    second code path that could disagree with what happened."""

    table = "events"

    def append(self, scope: TenantScope, session_id: str | None, name: str, payload) -> None:
        self._store.execute(
            "INSERT INTO events (org_id, session_id, name, payload, at) VALUES (?, ?, ?, ?, ?)",
            (scope.org_id, session_id, name, json.dumps(payload, default=str), self._now()),
        )

    def for_session(self, scope: TenantScope, session_id: str) -> list[Row]:
        return self._select(scope, "session_id = ?", [session_id], order="at, id")

    def purge_before(self, scope: TenantScope, cutoff: float) -> int:
        clause, params = self._where(scope, "at < ?")
        rows = self._store.query(f"SELECT COUNT(*) AS n FROM events WHERE {clause}", [*params, cutoff])
        self._store.execute(f"DELETE FROM events WHERE {clause}", [*params, cutoff])
        return int(rows[0]["n"])
