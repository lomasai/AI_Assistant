from __future__ import annotations

from lomas_store.repos.base import Repository, new_id
from lomas_store.scope import TenantScope
from lomas_store.store import Row


class AnswerRepo(Repository):
    """Answers are recorded per student, never per class."""

    table = "answers"

    def record(
        self,
        scope: TenantScope,
        session_id: str,
        student_id: str,
        question_ref: str,
        response: str | None,
        correct: bool | None,
        latency_ms: int | None,
    ) -> str:
        answer_id = new_id()
        self._store.execute(
            "INSERT INTO answers (id, org_id, session_id, student_id, question_ref, response,"
            " correct, latency_ms, answered_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (answer_id, scope.org_id, session_id, student_id, question_ref, response,
             None if correct is None else int(correct), latency_ms, self._now()),
        )
        return answer_id

    def for_session(self, scope: TenantScope, session_id: str) -> list[Row]:
        return self._select(scope, "session_id = ?", [session_id], order="answered_at")

    def for_student(self, scope: TenantScope, student_id: str) -> list[Row]:
        return self._select(scope, "student_id = ?", [student_id], order="answered_at")
