from __future__ import annotations

from lomas_store.repos.base import Repository, new_id
from lomas_store.scope import TenantScope
from lomas_store.store import Row

STATUS_RUNNING = "running"
STATUS_CLOSED = "closed"


class SessionRepo(Repository):
    table = "sessions"
    scoped_by = ("org_id", "class_id")

    def open(self, scope: TenantScope, language: str, topic: str | None, teacher: str | None) -> str:
        session_id = new_id()
        self._store.execute(
            "INSERT INTO sessions (id, org_id, school_id, class_id, teacher, topic, language,"
            " started_at, ended_at, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, ?)",
            (session_id, scope.org_id, scope.school_id, scope.class_id, teacher, topic,
             language, self._now(), STATUS_RUNNING),
        )
        return session_id

    def close(self, scope: TenantScope, session_id: str, status: str = STATUS_CLOSED) -> None:
        clause, params = self._where(scope, "id = ?")
        self._store.execute(
            f"UPDATE sessions SET ended_at = ?, status = ? WHERE {clause}",
            [self._now(), status, *params, session_id],
        )

    def get(self, scope: TenantScope, session_id: str) -> Row | None:
        return self._one(scope, "id = ?", [session_id])

    def recent(self, scope: TenantScope, limit: int) -> list[Row]:
        clause, params = self._where(scope)
        return self._store.query(
            f"SELECT * FROM sessions WHERE {clause} ORDER BY started_at DESC LIMIT ?",
            [*params, limit],
        )

    def mark_present(self, scope: TenantScope, session_id: str, student_id: str) -> None:
        self._store.execute(
            "INSERT OR IGNORE INTO session_students (id, org_id, session_id, student_id,"
            " joined_at, left_at, present) VALUES (?, ?, ?, ?, ?, NULL, 1)",
            (new_id(), scope.org_id, session_id, student_id, self._now()),
        )

    def mark_left(self, scope: TenantScope, session_id: str, student_id: str) -> None:
        self._store.execute(
            "UPDATE session_students SET left_at = ? WHERE org_id = ? AND session_id = ?"
            " AND student_id = ?",
            (self._now(), scope.org_id, session_id, student_id),
        )

    def roster(self, scope: TenantScope, session_id: str) -> list[Row]:
        return self._store.query(
            "SELECT ss.*, s.name, s.roll_no FROM session_students ss"
            " JOIN students s ON s.id = ss.student_id"
            " WHERE ss.org_id = ? AND ss.session_id = ? ORDER BY s.roll_no",
            (scope.org_id, session_id),
        )
