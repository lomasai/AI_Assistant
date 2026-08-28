from __future__ import annotations

import asyncio
import fnmatch
import threading
from typing import Any

from lomas_core import logging as log
from lomas_core.events import EventBus, to_plain
from lomas_core.schema import Config

ALL_EVENTS = "*"
EVENT_KEY = "event"
PAYLOAD_KEY = "payload"
PING = "ping"
HELLO = "hello"


class Client:
    """One browser's view of the stream.

    Bounded, drop-oldest. A tab left open on a locked laptop loses events; it
    does not slow the lesson down. That direction is not negotiable.
    """

    __slots__ = ("queue", "dropped")

    def __init__(self, size: int) -> None:
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=size)
        self.dropped = 0

    def offer(self, message: dict) -> None:
        if self.queue.full():
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            self.dropped += 1
        self.queue.put_nowait(message)


class EventHub:
    """Where the synchronous bus meets asyncio.

    Events arrive on whichever thread got there first - the flow, the vision
    pipeline, an agent - and browsers live on an event loop. This is the only
    place those two touch, so it is also the only place that has to be right
    about threads.
    """

    def __init__(self, bus: EventBus, cfg: Config) -> None:
        self.bus = bus
        self.cfg = cfg.web
        self.log = log.get("web")
        self.clients: set[Client] = set()
        self.delivered = 0

        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock = threading.Lock()
        bus.subscribe(ALL_EVENTS, self._on_event)

    def bind(self, loop: asyncio.AbstractEventLoop) -> None:
        """Called once the server is running. Until then events are simply
        not forwarded - there is nobody to forward them to."""
        self._loop = loop

    def register(self) -> Client:
        client = Client(self.cfg.client_queue)
        with self._lock:
            self.clients.add(client)
        return client

    def unregister(self, client: Client) -> None:
        with self._lock:
            self.clients.discard(client)

    def greeting(self) -> list[dict]:
        """What a browser opening mid-class needs before its first event: the
        recent past, so the face is not blank until somebody speaks."""
        return [_message(name, payload) for name, payload in self.bus.replay()]

    def wants(self, event: str) -> bool:
        return any(fnmatch.fnmatchcase(event, p) for p in self.cfg.event_filter)

    def _on_event(self, event: str, payload: Any) -> None:
        loop = self._loop
        if loop is None or not self.wants(event):
            return

        with self._lock:
            targets = list(self.clients)
        if not targets:
            return

        message = _message(event, payload)
        self.delivered += 1
        try:
            loop.call_soon_threadsafe(_deliver, targets, message)
        except RuntimeError:
            # The loop closed between the check and the call. A browser
            # missing an event while the server is shutting down is fine.
            self._loop = None


def _deliver(targets: list[Client], message: dict) -> None:
    for client in targets:
        client.offer(message)


def _message(event: str, payload: Any) -> dict:
    return {EVENT_KEY: event, PAYLOAD_KEY: to_plain(payload)}


async def pump(websocket, hub: EventHub) -> None:
    """Serve one browser until it goes away.

    The timeout is a heartbeat, not a deadline: a quiet classroom publishes
    nothing for minutes and a socket with no traffic gets closed by anything
    sitting between the robot and the tab.
    """
    client = hub.register()
    try:
        for message in hub.greeting():
            await websocket.send_json(message)

        while True:
            try:
                message = await asyncio.wait_for(client.queue.get(), hub.cfg.ping_seconds)
            except asyncio.TimeoutError:
                await websocket.send_json({EVENT_KEY: PING, PAYLOAD_KEY: None})
                continue
            await websocket.send_json(message)
    finally:
        hub.unregister(client)
