from __future__ import annotations

from lomas_store.repos.base import Repository, new_id
from lomas_store.scope import TenantScope
from lomas_store.store import Row


class CardRepo(Repository):
    """Which printed marker belongs to which child.

    A card is not a biometric: it is a number on a piece of paper, which is
    why a school that will not consent to storing a child's face can still
    have attendance, attribution and per-child marks. Cards are lost, swapped
    and re-issued, so this is a table of its own rather than a column on the
    child - the old card is withdrawn and the history stays readable.
    """

    table = "cards"
    scoped_by = ("org_id",)

    def issue(self, scope: TenantScope, student_id: str, marker_id: int) -> str:
        # One live card per marker. Handing the same number to two children
        # is how one child's answers end up against another's name.
        self.withdraw(scope, marker_id)
        card_id = new_id()
        self._store.execute(
            "INSERT INTO cards (id, org_id, student_id, marker_id, issued_at, withdrawn_at)"
            " VALUES (?, ?, ?, ?, ?, NULL)",
            (card_id, scope.org_id, student_id, marker_id, self._now()),
        )
        return card_id

    def withdraw(self, scope: TenantScope, marker_id: int) -> None:
        self._store.execute(
            "UPDATE cards SET withdrawn_at = ? WHERE org_id = ? AND marker_id = ?"
            " AND withdrawn_at IS NULL",
            (self._now(), scope.org_id, marker_id),
        )

    def owner(self, scope: TenantScope, marker_id: int) -> Row | None:
        rows = self._store.query(
            "SELECT c.*, s.name, s.roll_no FROM cards c JOIN students s ON s.id = c.student_id"
            " WHERE c.org_id = ? AND c.marker_id = ? AND c.withdrawn_at IS NULL",
            (scope.org_id, marker_id),
        )
        return rows[0] if rows else None

    def for_class(self, scope: TenantScope) -> list[Row]:
        return self._store.query(
            "SELECT c.*, s.name, s.roll_no FROM cards c JOIN students s ON s.id = c.student_id"
            " WHERE c.org_id = ? AND s.class_id = ? AND c.withdrawn_at IS NULL"
            " ORDER BY s.roll_no",
            (scope.org_id, scope.class_id),
        )

    def of_student(self, scope: TenantScope, student_id: str) -> Row | None:
        rows = self._store.query(
            "SELECT * FROM cards WHERE org_id = ? AND student_id = ? AND withdrawn_at IS NULL",
            (scope.org_id, student_id),
        )
        return rows[0] if rows else None
