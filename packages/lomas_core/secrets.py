from __future__ import annotations

import os
from pathlib import Path

# Secrets go into the environment, never into Config.
#
# That is the whole design. Config is dumped by --show-config and served at
# /api/debug/config, so anything living there is on a web page. What the
# config holds is the *name* of the variable; what this file loads is the
# value, and the two never meet.

COMMENT = "#"
ASSIGN = "="
QUOTES = "\"'"
SECRETS_FILE = "secrets.env"

# Settings are named after the config key they set and are safe to print.
# Everything else in the file is a key, and only its name may be logged.
SETTING_PREFIX = "LOMAS__"


def is_setting(name: str) -> bool:
    """A LOMAS__ variable names a config key, so it is a setting rather than
    a secret and may be printed in full."""
    return name.startswith(SETTING_PREFIX)


def load_secrets(path: str | Path, override: bool = False) -> list[str]:
    """Read KEY=value lines into the environment.

    Returns the names it set, never the values - a log line naming a key is
    a log line that has leaked one. An already-exported variable wins by
    default, so a real deployment can pass secrets in properly and still keep
    this file around for the bench.
    """
    file = Path(path)
    if not file.exists():
        return []

    loaded: list[str] = []
    for line in file.read_text(encoding="utf-8").splitlines():
        entry = line.strip()
        if not entry or entry.startswith(COMMENT) or ASSIGN not in entry:
            continue

        name, _, raw = entry.partition(ASSIGN)
        name = name.strip().removeprefix("export ").strip()
        value = raw.strip().strip(QUOTES)
        if not name or not value:
            continue
        if name in os.environ and not override:
            continue

        os.environ[name] = value
        loaded.append(name)

    return loaded
