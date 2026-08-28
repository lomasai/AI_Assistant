from __future__ import annotations

import asyncio
import fnmatch
import inspect
import threading
from collections import deque
from typing import Any, Callable

Handler = Callable[[str, Any], Any]
PLAIN = (str, int, float, bool)


def to_plain(payload: Any) -> Any:
    """A payload as plain data, for the log and for the wire.

    Contract payloads are slotted dataclasses and nest - a tracks event holds
    a tuple of track views - so this recurses. Anything unrecognised becomes
    its string form rather than raising: a serialiser that can stop the class
    is worse than a serialiser that is occasionally vague.
    """
    if payload is None or isinstance(payload, PLAIN):
        return payload
    if isinstance(payload, dict):
        return {str(k): to_plain(v) for k, v in payload.items()}
    if isinstance(payload, (list, tuple, set)):
        return [to_plain(item) for item in payload]
    slots = getattr(payload, "__slots__", None)
    if slots:
        return {slot: to_plain(getattr(payload, slot, None)) for slot in slots}
    return str(payload)


ErrorHook = Callable[[str, BaseException], None]


class Subscription:
    __slots__ = ("pattern", "handler")

    def __init__(self, pattern: str, handler: Handler) -> None:
        self.pattern = pattern
        self.handler = handler


class EventBus:
    """The only way features reach each other.

    Patterns are fnmatch style, so "student.*" and "*" both work. A handler
    that raises is routed to `on_error`; if no hook is set the exception
    propagates, which is what debug mode wants.
    """

    def __init__(
        self,
        replay_size: int,
        on_error: ErrorHook | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        self._subs: list[Subscription] = []
        self._replay: deque[tuple[str, Any]] = deque(maxlen=replay_size)
        self._lock = threading.RLock()
        self._on_error = on_error
        self._loop = loop

    def subscribe(self, pattern: str, handler: Handler) -> Callable[[], None]:
        sub = Subscription(pattern, handler)
        with self._lock:
            self._subs.append(sub)

        def unsubscribe() -> None:
            with self._lock:
                if sub in self._subs:
                    self._subs.remove(sub)

        return unsubscribe

    def publish(self, name: str, payload: Any = None) -> None:
        with self._lock:
            self._replay.append((name, payload))
            targets = [s.handler for s in self._subs if fnmatch.fnmatchcase(name, s.pattern)]

        for handler in targets:
            try:
                result = handler(name, payload)
                if inspect.isawaitable(result):
                    self._schedule(result)
            except Exception as exc:
                if self._on_error is None:
                    raise
                self._on_error(name, exc)

    def replay(self, pattern: str = "*") -> list[tuple[str, Any]]:
        with self._lock:
            return [(n, p) for n, p in self._replay if fnmatch.fnmatchcase(n, pattern)]

    def clear(self) -> None:
        with self._lock:
            self._subs.clear()
            self._replay.clear()

    def _schedule(self, awaitable: Any) -> None:
        loop = self._loop
        if loop is None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                awaitable.close()
                raise RuntimeError(
                    "async event handler registered but no event loop is running; "
                    "pass loop= to EventBus or use a sync handler"
                ) from None
            loop.create_task(awaitable)
            return
        asyncio.run_coroutine_threadsafe(awaitable, loop)
