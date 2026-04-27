"""Server package initialization.

Loads project-local environment variables before module-level service
singletons read their configuration.
"""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    if ENV_PATH.exists():
        for raw_line in ENV_PATH.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))
else:
    load_dotenv(ENV_PATH)
