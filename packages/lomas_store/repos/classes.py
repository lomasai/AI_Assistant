from __future__ import annotations

from lomas_store.repos.base import Repository, new_id
from lomas_store.scope import TenantScope
from lomas_store.store import Row


class ClassRepo(Repository):
    table = "classes"
    scoped_by = ("org_id", "school_id")

    def create(self, scope: TenantScope, grade: str, section: str, subject: str) -> str:
        class_id = scope.class_id or new_id()
        self._store.execute(
            "INSERT INTO classes (id, org_id, school_id, grade, section, subject, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (class_id, scope.org_id, scope.school_id, grade, section, subject, self._now()),
        )
        return class_id

    def get(self, scope: TenantScope, class_id: str) -> Row | None:
        return self._one(scope, "id = ?", [class_id])

    def list_for_school(self, scope: TenantScope) -> list[Row]:
        return self._select(scope, order="grade, section")
