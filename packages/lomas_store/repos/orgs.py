from __future__ import annotations

from lomas_store.repos.base import Repository, new_id
from lomas_store.scope import TenantScope
from lomas_store.store import Row


class OrgRepo(Repository):
    table = "orgs"
    columns = {"org_id": "id"}

    def create(self, scope: TenantScope, name: str) -> str:
        self._store.execute(
            "INSERT INTO orgs (id, name, created_at) VALUES (?, ?, ?)",
            (scope.org_id, name, self._now()),
        )
        return scope.org_id

    def get(self, scope: TenantScope) -> Row | None:
        rows = self._select(scope)
        return rows[0] if rows else None

    def ensure(self, scope: TenantScope, name: str) -> str:
        existing = self.get(scope)
        if existing is not None:
            return str(existing["id"])
        return self.create(scope, name)
