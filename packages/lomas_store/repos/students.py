from __future__ import annotations

from lomas_store.repos.base import Repository, new_id
from lomas_store.scope import TenantScope
from lomas_store.store import Row


class StudentRepo(Repository):
    table = "students"
    scoped_by = ("org_id", "class_id")

    def create(self, scope: TenantScope, name: str, roll_no: str) -> str:
        student_id = new_id()
        self._store.execute(
            "INSERT INTO students (id, org_id, school_id, class_id, name, roll_no, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (student_id, scope.org_id, scope.school_id, scope.class_id, name, roll_no, self._now()),
        )
        return student_id

    def get(self, scope: TenantScope, student_id: str) -> Row | None:
        return self._one(scope, "id = ?", [student_id])

    def by_roll(self, scope: TenantScope, roll_no: str) -> Row | None:
        return self._one(scope, "roll_no = ?", [roll_no])

    def list_for_class(self, scope: TenantScope) -> list[Row]:
        return self._select(scope, order="roll_no")

    def delete(self, scope: TenantScope, student_id: str) -> None:
        clause, params = self._where(scope, "id = ?")
        self._store.execute(f"DELETE FROM students WHERE {clause}", [*params, student_id])
