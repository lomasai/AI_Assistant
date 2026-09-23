from __future__ import annotations

from lomas_core.schema import SyncConfig

from app.sync import SYNCS

NONE = "none"
NOWHERE = "nowhere"


@SYNCS.register(NONE)
class NoFiler:
    """Files nothing, on purpose.

    A school whose robots are not in a repository, or whose network policy
    says nothing leaves the building, sets this rather than switching the
    whole feature off - so `sync.enabled` stays true and the logs still say
    what would have been filed.
    """

    def __init__(self, cfg: SyncConfig) -> None:
        self.cfg = cfg

    @property
    def available(self) -> bool:
        return True

    def describe(self) -> str:
        return NOWHERE

    def send(self, paths: list[str], message: str) -> str:
        return ""
