from __future__ import annotations

import importlib
import pkgutil
from typing import Callable, Generic, TypeVar

from lomas_core.errors import RegistryError

T = TypeVar("T")


class Registry(Generic[T]):
    """Maps a config string to an implementation class.

    Every pluggable family in the system uses one of these. Implementations
    register themselves on import; `discover` imports a package's modules so
    that happens without anyone maintaining a list.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._entries: dict[str, type[T]] = {}
        self._discovered: set[str] = set()

    def register(self, key: str) -> Callable[[type[T]], type[T]]:
        def decorator(cls: type[T]) -> type[T]:
            claimed = self._entries.get(key)
            if claimed is not None and claimed is not cls:
                raise RegistryError(
                    f"{self.name}: '{key}' is already registered to {claimed.__name__}"
                )
            self._entries[key] = cls
            return cls

        return decorator

    def get(self, key: str) -> type[T]:
        try:
            return self._entries[key]
        except KeyError:
            known = ", ".join(self.keys()) or "none registered"
            raise RegistryError(f"{self.name}: no implementation '{key}'. Known: {known}") from None

    def create(self, key: str, *args, **kwargs) -> T:
        return self.get(key)(*args, **kwargs)

    def keys(self) -> list[str]:
        return sorted(self._entries)

    def discover(self, package: str) -> None:
        """Import every module in a package so its decorators run."""
        if package in self._discovered:
            return
        module = importlib.import_module(package)
        for found in pkgutil.iter_modules(module.__path__):
            importlib.import_module(f"{package}.{found.name}")
        self._discovered.add(package)

    def __contains__(self, key: object) -> bool:
        return key in self._entries
