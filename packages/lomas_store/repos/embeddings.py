from __future__ import annotations

from lomas_store.repos.base import Repository, new_id
from lomas_store.scope import TenantScope
from lomas_store.store import Row


class EmbeddingRepo(Repository):
    """Stores face vectors as raw bytes.

    `dim` and `dtype` travel with the blob so a caller can rebuild the array
    without this package depending on numpy.
    """

    table = "embeddings"

    def add(
        self,
        scope: TenantScope,
        student_id: str,
        vector: bytes,
        dim: int,
        dtype: str,
        quality: float,
        angle: str,
    ) -> str:
        embedding_id = new_id()
        self._store.execute(
            "INSERT INTO embeddings (id, org_id, student_id, vector, dim, dtype, quality,"
            " angle, captured_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (embedding_id, scope.org_id, student_id, vector, dim, dtype, quality, angle, self._now()),
        )
        return embedding_id

    def for_student(self, scope: TenantScope, student_id: str) -> list[Row]:
        return self._select(scope, "student_id = ?", [student_id], order="captured_at")

    def all_for_class(self, scope: TenantScope) -> list[Row]:
        org_id, class_id = scope.require("org_id", "class_id")
        return self._store.query(
            "SELECT e.* FROM embeddings e JOIN students s ON s.id = e.student_id"
            " WHERE e.org_id = ? AND s.org_id = ? AND s.class_id = ?",
            (org_id, org_id, class_id),
        )

    def delete_for_student(self, scope: TenantScope, student_id: str) -> None:
        clause, params = self._where(scope, "student_id = ?")
        self._store.execute(f"DELETE FROM embeddings WHERE {clause}", [*params, student_id])
