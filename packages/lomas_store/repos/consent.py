from __future__ import annotations

from lomas_store.repos.base import Repository, new_id
from lomas_store.scope import TenantScope
from lomas_store.store import Row


class ConsentRepo(Repository):
    """Enrolment is blocked unless `is_granted` returns True.

    The check lives in the data layer rather than the UI so that a future
    endpoint cannot forget it.
    """

    table = "consent"

    def grant(
        self,
        scope: TenantScope,
        student_id: str,
        kind: str,
        granted_by: str,
        document_ref: str | None = None,
    ) -> str:
        consent_id = new_id()
        self._store.execute(
            "INSERT INTO consent (id, org_id, student_id, kind, granted_by, granted_at,"
            " document_ref, revoked_at) VALUES (?, ?, ?, ?, ?, ?, ?, NULL)",
            (consent_id, scope.org_id, student_id, kind, granted_by, self._now(), document_ref),
        )
        return consent_id

    def revoke(self, scope: TenantScope, student_id: str, kind: str) -> None:
        clause, params = self._where(scope, "student_id = ? AND kind = ?")
        self._store.execute(
            f"UPDATE consent SET revoked_at = ? WHERE {clause}",
            [self._now(), *params, student_id, kind],
        )

    def is_granted(self, scope: TenantScope, student_id: str, kind: str) -> bool:
        row = self._one(
            scope, "student_id = ? AND kind = ? AND revoked_at IS NULL", [student_id, kind]
        )
        return row is not None

    def for_student(self, scope: TenantScope, student_id: str) -> list[Row]:
        return self._select(scope, "student_id = ?", [student_id], order="granted_at")
