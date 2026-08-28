from __future__ import annotations

from dataclasses import dataclass

from lomas_core.errors import LomasError


class ScopeError(LomasError):
    """A repository was called without the tenant fields it needs."""


@dataclass(frozen=True, slots=True)
class TenantScope:
    """Who is asking. Every repository method takes one of these first.

    There is no way to reach the data layer without saying which tenant you
    are, because the signature will not let you.
    """

    org_id: str
    school_id: str | None = None
    class_id: str | None = None

    def require(self, *fields: str) -> tuple[str, ...]:
        missing = [f for f in fields if getattr(self, f) is None]
        if missing:
            raise ScopeError(f"scope is missing {', '.join(missing)}")
        return tuple(getattr(self, f) for f in fields)

    def narrowed(self, **fields: str | None) -> TenantScope:
        return TenantScope(
            org_id=fields.get("org_id") or self.org_id,
            school_id=fields.get("school_id") or self.school_id,
            class_id=fields.get("class_id") or self.class_id,
        )
