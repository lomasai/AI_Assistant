from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any, Iterable, Protocol, Sequence, runtime_checkable

from lomas_core.registry import Registry

Row = dict[str, Any]


@runtime_checkable
class Store(Protocol):
    def execute(self, sql: str, params: Sequence[Any] = ()) -> None: ...

    def executemany(self, sql: str, rows: Iterable[Sequence[Any]]) -> None: ...

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[Row]: ...

    def transaction(self) -> AbstractContextManager[None]: ...

    def close(self) -> None: ...


STORES: Registry[Store] = Registry("storage backend")
