#!/usr/bin/env python3
"""Single entry point. Parses arguments, loads config, runs a session."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "packages"))
sys.path.insert(0, str(ROOT))

from lomas_core import logging as log  # noqa: E402
from lomas_core.config import load  # noqa: E402
from lomas_core.errors import LomasError  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="lomasai", description="LomasAI classroom system")
    parser.add_argument("--mode", default="user", help="config profile to run (debug, user, pi)")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override any config key, e.g. --set llm.provider=anthropic",
    )
    parser.add_argument("--config-dir", default=str(ROOT / "config"))
    parser.add_argument("--topic", default="", help="lesson to teach (default from config)")
    parser.add_argument("--language", default="", help="language to teach in")
    parser.add_argument("--teacher", default="", help="name recorded against the session")
    parser.add_argument("--show-config", action="store_true", help="print resolved config and exit")
    parser.add_argument("--list-plugins", action="store_true", help="print what config can select")
    parser.add_argument("--seed", action="store_true", help="create a demo class if none exists")
    return parser.parse_args(argv)


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
        print(json.dumps(cfg.model_dump(), indent=2, sort_keys=True))
        return 0

    from app import container, seed

    if args.list_plugins:
        print(json.dumps(container.available(), indent=2, sort_keys=True))
        return 0

    system = container.build(cfg)
    try:
        if args.seed:
            seed.demo_class(system)

        logger.info("LomasAI, %s mode, org %s", cfg.runtime.mode, cfg.active_org_id)
        state = system.orchestrator.run(topic=args.topic, language=args.language)
        logger.info("finished: %s", state.value)
        return 0
    except LomasError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        system.close()


if __name__ == "__main__":
    raise SystemExit(main())
