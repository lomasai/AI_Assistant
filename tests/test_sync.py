"""The robot filing its own traces.

Every fix to the audio, the latency and the face came out of a trace pushed
from the Pi by hand: three git commands, after every class, typed on a robot
that does not have a keyboard in front of it. Traces that need that are
traces that stop arriving.
"""
from __future__ import annotations

import time

import pytest

from lomas_core.clock import FakeClock, RealClock
from lomas_core.config import load
from lomas_core.contracts import SESSION_CLOSED, SESSION_PAUSED, SessionClosed
from app import container
from app.sync import SYNCS, FileSync
from app.syncs.git import GitFiler


@SYNCS.register("remembers")
class Remembers:
    """A filer that writes nothing anywhere and says what it was asked to."""

    sent: list[tuple[tuple[str, ...], str]] = []
    slow = 0.0   # a real push takes a moment; some tests need it to

    def __init__(self, cfg) -> None:
        self.cfg = cfg

    @property
    def available(self) -> bool:
        return True

    def describe(self) -> str:
        return "a list"

    def send(self, paths, message: str) -> str:
        time.sleep(Remembers.slow)
        Remembers.sent.append((tuple(paths), message))
        return message


def build(*extra: str, clock=None):
    cfg = load("config", "debug", ["sync.enabled=true", "sync.backend=remembers", *extra],
               use_env=False)
    Remembers.sent = []
    Remembers.slow = 0.0
    return FileSync(cfg, container.event_bus(cfg), clock or RealClock())


def settle(wanted: int = 1, seconds: float = 2.0) -> None:
    """Filing happens off the class's thread, so a test waits for it."""
    deadline = time.monotonic() + seconds
    while len(Remembers.sent) < wanted and time.monotonic() < deadline:
        time.sleep(0.01)


# --- when it files ---------------------------------------------------------


def test_the_end_of_a_class_files_the_trace() -> None:
    sync = build()

    sync.bus.publish(SESSION_CLOSED, SessionClosed(session_id="s1", ended_at=1.0, reason="closed"))
    settle()

    assert len(Remembers.sent) == 1
    assert "session_closed" in Remembers.sent[0][1]
    sync.close()


def test_which_moments_file_is_config() -> None:
    sync = build("sync.on=[paused]")

    sync.bus.publish(SESSION_CLOSED, SessionClosed(session_id="s1", ended_at=1.0, reason="closed"))
    settle(seconds=0.3)
    assert Remembers.sent == [], "it filed at a moment nobody asked for"

    sync.bus.publish(SESSION_PAUSED, {"session_id": "s1"})
    settle()
    assert len(Remembers.sent) == 1
    sync.close()


def test_switching_off_files_what_is_there() -> None:
    sync = build()

    sync.flush()

    assert len(Remembers.sent) == 1
    assert "shutdown" in Remembers.sent[0][1]
    sync.close()


def test_shutdown_can_be_left_out() -> None:
    sync = build("sync.on=[session_closed]")

    sync.flush()

    assert Remembers.sent == []
    sync.close()


def test_nothing_happens_when_it_is_off() -> None:
    cfg = load("config", "debug", [], use_env=False)
    sync = FileSync(cfg, container.event_bus(cfg), FakeClock())

    sync.file("session_closed")
    sync.flush()

    assert not sync.enabled
    assert Remembers.sent == []


def test_only_the_paths_it_was_given() -> None:
    """A robot that commits its whole working tree commits whatever
    half-finished edit somebody left on it."""
    sync = build("sync.paths=[data/logs,data/lessons]")

    sync.file("session_closed")
    settle()

    assert Remembers.sent[0][0] == ("data/logs", "data/lessons")
    sync.close()


def test_two_classes_ending_at_once_file_once() -> None:
    sync = build()
    Remembers.slow = 0.2

    for _ in range(4):
        sync.file("session_closed")
    settle(seconds=3.0)
    time.sleep(0.3)

    assert len(Remembers.sent) < 4, "four pushes of the same files raced each other"
    sync.close()


def test_a_backend_that_throws_does_not_end_a_class() -> None:
    sync = build()
    sync.backend.send = lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("no network"))

    sync.file("session_closed")
    time.sleep(0.2)

    assert sync.filed == 0, "a failed push was counted as a filed trace"
    sync.close()


# --- what git is asked to do ----------------------------------------------


@pytest.fixture
def git_cfg():
    return load("config", "debug", [], use_env=False).sync


def test_git_never_commits_everything(git_cfg) -> None:
    filer = GitFiler(git_cfg)
    ran: list[list[str]] = []
    filer._git = lambda argv: (ran.append(argv), (True, ""))[1]

    filer.send(["data/logs"], "trace: test")

    assert ran[0] == ["add", "--", "data/logs"]
    assert not any("-A" in argv or "--all" in argv for argv in ran)
    assert ran[1][:2] == ["commit", "-m"]


def test_nothing_new_is_not_a_failure(git_cfg) -> None:
    filer = GitFiler(git_cfg)
    filer._git = lambda argv: (False, "nothing to commit, working tree clean")

    assert filer.send(["data/logs"], "trace: test") == ""


def test_a_commit_that_cannot_be_pushed_is_still_a_commit(git_cfg) -> None:
    """A school with no internet at four o'clock keeps the commit, and the
    next class pushes it."""
    filer = GitFiler(git_cfg)
    replies = {"add": (True, ""), "commit": (True, "1 file changed"),
               "push": (False, "could not resolve host")}
    filer._git = lambda argv: replies[argv[0]]

    said = filer.send(["data/logs"], "trace: test")

    assert "not pushed" in said


def test_a_robot_with_no_network_can_be_told_not_to_push() -> None:
    cfg = load("config", "debug", ["sync.push=false"], use_env=False).sync
    filer = GitFiler(cfg)
    filer._git = lambda argv: (True, "1 file changed")

    assert "locally" in filer.send(["data/logs"], "trace: test")


def test_paths_that_are_not_there_are_skipped(git_cfg) -> None:
    filer = GitFiler(git_cfg)
    filer._git = lambda argv: (True, "")

    assert filer.send(["data/nothing-here-at-all"], "trace: test") == ""
