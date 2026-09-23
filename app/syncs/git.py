from __future__ import annotations

import subprocess
from pathlib import Path

from lomas_core import logging as log
from lomas_core.schema import SyncConfig

from app.sync import SYNCS

GIT = "git"
INSIDE = ["rev-parse", "--is-inside-work-tree"]
TRUE = "true"
NOTHING = ("nothing to commit", "no changes added")
FAILED = 200  # how much of git's complaint is worth keeping


@SYNCS.register(GIT)
class GitFiler:
    """Commits and pushes the paths it was given, and nothing else.

    The repository is already how this code reaches the robot, so it is also
    the way back: a trace pushed from the Pi is readable from anywhere a few
    seconds later, with no new service, no credentials to hand out and no
    upload endpoint to keep running.

    Never `git add -A`. A robot that commits its whole working tree commits
    whatever half-finished edit somebody left on it at four o'clock.
    """

    def __init__(self, cfg: SyncConfig) -> None:
        self.cfg = cfg
        self.log = log.get("sync")

    @property
    def available(self) -> bool:
        worked, reply = self._git(INSIDE)
        return worked and reply.strip() == TRUE

    def describe(self) -> str:
        where = f"{self.cfg.remote}/{self.cfg.branch}" if self.cfg.branch else self.cfg.remote
        return where if self.cfg.push else "this machine only"

    def send(self, paths: list[str], message: str) -> str:
        here = [path for path in paths if Path(path).exists()]
        if not here:
            return ""

        self._git(["add", "--", *here])
        worked, reply = self._git(["commit", "-m", message, "--", *here])
        if not worked:
            # Nothing new since the last class is the usual reason, and it is
            # not a failure. Anything else has already reached the log.
            return ""

        if not self.cfg.push:
            return f"filed locally: {message}"

        pushed, complaint = self._git(self._push())
        if not pushed:
            # The commit is made and the next class will push it, so this is
            # a note rather than a problem.
            return f"committed but not pushed: {complaint.splitlines()[0][:FAILED]}"
        return f"pushed {', '.join(here)}: {message}"

    def _push(self) -> list[str]:
        argv = ["push", self.cfg.remote]
        if self.cfg.branch:
            argv.append(self.cfg.branch)
        return argv

    def _git(self, argv: list[str]) -> tuple[bool, str]:
        """Both streams, because git says useful things on each - and a
        failed push has to reach the log rather than an exception nobody sees
        on a robot with no screen."""
        try:
            done = subprocess.run([GIT, *argv], capture_output=True, text=True,
                                  timeout=self.cfg.timeout_seconds, check=False)
        except (OSError, subprocess.SubprocessError) as exc:
            return False, str(exc)

        reply = f"{done.stdout}{done.stderr}".strip()
        if done.returncode:
            if not any(quiet in reply for quiet in NOTHING):
                self.log.debug("git %s said: %s", argv[0], reply[:FAILED])
            return False, reply
        return True, reply
