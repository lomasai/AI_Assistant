from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from lomas_core.schema import RuntimeConfig

LOGGER_ROOT = "lomas"


class HumanFormatter(logging.Formatter):
    """User mode: one readable line, no machinery."""

    def format(self, record: logging.LogRecord) -> str:
        return f"{record.getMessage()}"


class DebugFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        stamp = self.formatTime(record, "%H:%M:%S")
        where = record.name.removeprefix(f"{LOGGER_ROOT}.")
        line = f"{stamp} {record.levelname[0]} {where:<22} {record.getMessage()}"
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


class JsonlFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": record.created,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure(cfg: RuntimeConfig) -> logging.Logger:
    root = logging.getLogger(LOGGER_ROOT)
    root.handlers.clear()
    root.setLevel(cfg.log_level)
    root.propagate = False

    if "console" in cfg.sinks:
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(DebugFormatter() if cfg.mode == "debug" else HumanFormatter())
        root.addHandler(console)

    if "jsonl" in cfg.sinks:
        directory = Path(cfg.log_dir)
        directory.mkdir(parents=True, exist_ok=True)
        stream = logging.FileHandler(directory / "lomas.jsonl", encoding="utf-8")
        stream.setFormatter(JsonlFormatter())
        root.addHandler(stream)

    return root


def get(name: str) -> logging.Logger:
    return logging.getLogger(f"{LOGGER_ROOT}.{name}")
