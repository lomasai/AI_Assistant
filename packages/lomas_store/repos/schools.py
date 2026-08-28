from __future__ import annotations

from lomas_store.repos.base import Repository, new_id
from lomas_store.scope import TenantScope
from lomas_store.store import Row


class SchoolRepo(Repository):
    table = "schools"

    def create(self, scope: TenantScope, name: str, default_language: str) -> str:
        school_id = scope.school_id or new_id()
        self._store.execute(
            "INSERT INTO schools (id, org_id, name, default_language, created_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (school_id, scope.org_id, name, default_language, self._now()),
        )
        return school_id

    def get(self, scope: TenantScope, school_id: str) -> Row | None:
        return self._one(scope, "id = ?", [school_id])

    def list_for_org(self, scope: TenantScope) -> list[Row]:
        return self._select(scope, order="name")
