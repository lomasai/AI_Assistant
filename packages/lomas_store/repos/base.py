from __future__ import annotations

import time
import uuid
from typing import Any, Callable, ClassVar

from lomas_store.scope import TenantScope
from lomas_store.store import Row, Store


def new_id() -> str:
    return uuid.uuid4().hex


class Repository:
    """Base for every repository.

    `scoped_by` declares which tenant fields this table is filtered on. A
    scope missing one of them raises immediately rather than quietly widening
    the query, which is the failure mode that leaks one school's data into
    another's.
    """

    table: ClassVar[str]
    scoped_by: ClassVar[tuple[str, ...]] = ("org_id",)
    # Only needed where the column name differs from the scope field, as on
    # the orgs table where the tenant is the primary key.
    columns: ClassVar[dict[str, str]] = {}

    def __init__(self, store: Store, now: Callable[[], float] = time.time) -> None:
        self._store = store
        self._now = now

    def _where(self, scope: TenantScope, extra: str = "") -> tuple[str, list[Any]]:
        values = scope.require(*self.scoped_by)
        clause = " AND ".join(f"{self.columns.get(f, f)} = ?" for f in self.scoped_by)
        params = list(values)
        if extra:
            clause = f"{clause} AND {extra}"
        return clause, params

    def _select(self, scope: TenantScope, extra: str = "", params: list[Any] | None = None,
                order: str = "") -> list[Row]:
        clause, values = self._where(scope, extra)
        values.extend(params or [])
        sql = f"SELECT * FROM {self.table} WHERE {clause}"
        if order:
            sql = f"{sql} ORDER BY {order}"
        return self._store.query(sql, values)

    def _one(self, scope: TenantScope, extra: str, params: list[Any]) -> Row | None:
        rows = self._select(scope, extra, params)
        return rows[0] if rows else None
