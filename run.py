#!/usr/bin/env python3
"""Single entry point. Parses arguments, loads config, runs a session."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WAIT_POLL_SECONDS = 0.5
sys.path.insert(0, str(ROOT / "packages"))
sys.path.insert(0, str(ROOT))

from lomas_core import logging as log  # noqa: E402
from lomas_core.config import load  # noqa: E402
from lomas_core.secrets import SECRETS_FILE, is_setting, load_secrets  # noqa: E402
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
    parser.add_argument(
        "--secrets",
        default="",
        help=f"file of KEY=value secrets (default: <config-dir>/{SECRETS_FILE})",
    )
    parser.add_argument("--topic", default="", help="lesson to teach (default from config)")
    parser.add_argument("--language", default="", help="language to teach in")
    parser.add_argument("--teacher", default="", help="name recorded against the session")
    parser.add_argument("--show-config", action="store_true", help="print resolved config and exit")
    parser.add_argument("--list-plugins", action="store_true", help="print what config can select")
    parser.add_argument("--seed", action="store_true", help="create a demo class if none exists")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Before the config, because the config only ever holds the *name* of a
    # key. An already-exported variable wins.
    keys = load_secrets(args.secrets or Path(args.config_dir) / SECRETS_FILE)

    try:
        cfg = load(args.config_dir, args.mode, args.overrides)
    except LomasError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    log.configure(cfg.runtime)
    logger = log.get("run")
    if keys:
        # Settings are named after the config key they set, so they are safe
        # to print. A key is not: a log line carrying one has leaked it, so
        # only the count is shown.
        settings = [name for name in keys if is_setting(name)]
        secret_count = len(keys) - len(settings)
        logger.info(
            "%s: %d key(s), %d setting(s)%s",
            args.secrets or Path(args.config_dir) / SECRETS_FILE,
            secret_count,
            len(settings),
            f" [{', '.join(settings)}]" if settings else "",
        )

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

        # A missing body or a missing camera is not a missing lesson. Both
        # degrade: the robot teaches without moving, and attendance falls back
        # to the roster. A loose serial cable must not stop a class.
        if system.body is not None:
            optional(logger, "body", system.body.start)

        # The camera comes up with the robot; recognition waits for a class.
        if system.vision is not None:
            optional(logger, "camera", system.vision.watch)

        if system.web is not None:
            system.web.start()

        logger.info("LomasAI, %s mode, org %s", cfg.runtime.mode, cfg.active_org_id)

        if not cfg.flow.autostart:
            return wait(system, logger)

        state = system.orchestrator.run(topic=args.topic, language=args.language)
        logger.info("finished: %s", state.value)
        return 0
    except LomasError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        system.close()


def optional(logger, what: str, start) -> None:
    """Start something the robot is better with and can teach without.

    The web server and the flow are not in here on purpose: without those
    there is no product, and failing loudly is right. A body and a camera are
    different - a school with the ESP32 unplugged should still get a lesson.
    """
    try:
        start()
    except LomasError as exc:
        logger.error("%s unavailable, continuing without it: %s", what, exc)
    except Exception as exc:  # a driver, not us
        logger.error("%s failed to start, continuing without it: %s", what, exc)


def wait(system, logger) -> int:
    """Boot, show a sleeping face, and wait to be told to begin.

    This is what a robot standing in a classroom does. The bench runs one
    class and exits instead, which is flow.autostart.
    """
    if system.web is None:
        logger.error("flow.autostart is off and there are no surfaces; nothing to wait for")
        return 2

    logger.info("waiting for a class. Open %s and press Start.", system.web.url)
    try:
        while True:
            time.sleep(WAIT_POLL_SECONDS)
    except KeyboardInterrupt:
        logger.info("stopping")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
