#!/usr/bin/env python3
"""Single entry point. Parses arguments, loads config, starts the system."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "packages"))

from lomas_core import logging as log  # noqa: E402
from lomas_core.config import load  # noqa: E402
from lomas_core.errors import LomasError  # noqa: E402
from lomas_core.events import EventBus  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="lomasai", description="LomasAI classroom system")
    parser.add_argument("--mode", default="user", help="config profile to run (debug, user, pi)")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override any config key, e.g. --set runtime.log_level=DEBUG",
    )
    parser.add_argument("--config-dir", default=str(ROOT / "config"))
    parser.add_argument("--show-config", action="store_true", help="print resolved config and exit")
    return parser.parse_args(argv)


def build_bus(cfg) -> EventBus:
    logger = log.get("events")

    def on_error(event: str, exc: BaseException) -> None:
        logger.error("handler failed on %s: %s", event, exc, exc_info=exc)

    return EventBus(
        replay_size=cfg.runtime.event_replay_size,
        on_error=None if cfg.runtime.raise_on_handler_error else on_error,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        cfg = load(args.config_dir, args.mode, args.overrides)
    except LomasError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    log.configure(cfg.runtime)
    logger = log.get("run")

    if args.show_config:
        import json

        print(json.dumps(cfg.model_dump(), indent=2, sort_keys=True))
        return 0

    bus = build_bus(cfg)
    logger.info("LomasAI starting in %s mode, org %s", cfg.runtime.mode, cfg.active_org_id)
    logger.debug("event replay buffer holds %d events", cfg.runtime.event_replay_size)

    # P1 onward replaces this with the container and the orchestrator.
    logger.info("core is up. No subsystems are wired yet.")
    del bus
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
