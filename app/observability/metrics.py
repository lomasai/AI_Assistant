from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any

from lomas_core.clock import Clock
from lomas_core.events import EventBus, to_plain
from lomas_core.schema import Config

from app.observability.host import Host

ALL_EVENTS = "*"
MILLISECONDS = 1000.0
NOTHING = 0.0


class Window:
    """A rolling count, so a rate can be read without anyone keeping a timer.

    Everything here is measured from the outside. Diagnostics subscribe; they
    never take part, and nothing in this file can change what the robot does.
    """

    __slots__ = ("stamps", "seconds")

    def __init__(self, seconds: float, size: int) -> None:
        self.stamps: deque[float] = deque(maxlen=size)
        self.seconds = seconds

    def mark(self, at: float) -> None:
        self.stamps.append(at)

    def rate(self, now: float) -> float:
        recent = [s for s in self.stamps if now - s <= self.seconds]
        if len(recent) < 2:
            return NOTHING
        spread = recent[-1] - recent[0]
        return round((len(recent) - 1) / spread, 1) if spread > 0 else NOTHING


class Latency:
    """Milliseconds between one event and its answer.

    The pairs are config, so wiring a new one up - a wake word to its
    transcript, once there is a microphone loop - is a line in a file rather
    than a change here.
    """

    __slots__ = ("started", "samples", "keep")

    def __init__(self, keep: int) -> None:
        self.started: float | None = None
        self.samples: deque[float] = deque(maxlen=keep)
        self.keep = keep

    def begin(self, at: float) -> None:
        self.started = at

    def end(self, at: float) -> None:
        if self.started is None:
            return
        self.samples.append((at - self.started) * MILLISECONDS)
        self.started = None

    def summary(self) -> dict:
        if not self.samples:
            return {"count": 0, "last_ms": None, "mean_ms": None, "worst_ms": None}
        return {
            "count": len(self.samples),
            "last_ms": round(self.samples[-1], 1),
            "mean_ms": round(sum(self.samples) / len(self.samples), 1),
            "worst_ms": round(max(self.samples), 1),
        }


class LlmTap:
    """Wraps a provider to record what was actually sent.

    Installed only in debug mode, so in a classroom there is no prompt in
    memory and nothing to leak. It forwards every call untouched - a tap that
    can change an answer is not a tap.
    """

    def __init__(self, inner: Any, keep: int) -> None:
        self.inner = inner
        self.name = getattr(inner, "name", "")
        self.calls: deque[dict] = deque(maxlen=keep)

    def complete(self, messages, **options):
        began = time.perf_counter()
        answer = self.inner.complete(messages, **options)
        self._record(messages, answer, time.perf_counter() - began)
        return answer

    def stream(self, messages, **options):
        return self.inner.stream(messages, **options)

    def _record(self, messages, answer, seconds: float) -> None:
        usage = getattr(answer, "usage", None)
        self.calls.append(
            {
                "at": time.time(),
                "provider": getattr(answer, "provider", self.name),
                "model": getattr(answer, "model", ""),
                "ms": round(seconds * MILLISECONDS, 1),
                "prompt": [{"role": m.role, "content": m.content} for m in messages],
                "answer": getattr(answer, "text", ""),
                "input_tokens": getattr(usage, "input_tokens", 0),
                "output_tokens": getattr(usage, "output_tokens", 0),
            }
        )


class Metrics:
    """Everything the overlay shows, gathered from outside the features.

    It holds no reference that lets it call into the flow, and the only
    subscription is a read. Turn it off and nothing changes but the numbers.
    """

    def __init__(self, cfg: Config, bus: EventBus, clock: Clock) -> None:
        self.cfg = cfg
        self.debug = cfg.debug
        self.bus = bus
        self.clock = clock
        self.host = Host()
        self.started = time.time()
        self._lock = threading.RLock()

        self.counts: dict[str, int] = {}
        self.rates = {
            name: Window(self.debug.rate_window_seconds, self.debug.rate_samples)
            for name in self.debug.rate_events
        }
        self.latencies = {name: Latency(self.debug.keep_samples) for name in self.debug.latencies}
        self.recent: deque[dict] = deque(maxlen=self.debug.keep_events)
        self.taps: dict[str, LlmTap] = {}

        bus.subscribe(ALL_EVENTS, self._on_event)

    def tap(self, name: str, provider: Any) -> Any:
        """Wrap one provider and remember it. Called by the container, and
        only in debug mode - user mode never builds one of these."""
        tapped = LlmTap(provider, self.debug.keep_samples)
        self.taps[name] = tapped
        return tapped

    def snapshot(self, system) -> dict:
        now = time.monotonic()
        with self._lock:
            counts = dict(self.counts)
            rates = {name: window.rate(now) for name, window in self.rates.items()}
            latencies = {name: gauge.summary() for name, gauge in self.latencies.items()}
            recent = list(self.recent)

        return {
            "mode": self.cfg.runtime.mode,
            "uptime_s": round(time.time() - self.started, 1),
            "events": {"counts": counts, "rates": rates, "recent": recent},
            "latency": latencies,
            "vision": self._vision(system),
            "tracks": self._tracks(),
            "llm": self._llm(),
            "host": self.host.snapshot(),
            "config": {
                "detect_fps": self.cfg.face.detect_fps,
                "downscale_width": self.cfg.face.downscale_width,
                "provider": self.cfg.llm.provider,
                "tts": self.cfg.speech.tts.engine,
                "stt": self.cfg.speech.stt.engine,
                "wake": self.cfg.speech.wake.engine,
                "recognition": self.cfg.privacy.recognition_enabled,
            },
        }

    # --- panels -----------------------------------------------------------

    def _vision(self, system) -> dict:
        pipeline = getattr(system, "vision", None)
        if pipeline is None:
            return {"running": False}

        stats = dict(pipeline.stats())
        frames = pipeline.frames.stats() if pipeline.frames else {}
        stats["capture"] = frames
        stats["source"] = pipeline.source_id
        return stats

    def _tracks(self) -> list[dict]:
        """The last thing vision published, not a second copy of the tracker.
        Two answers to who is in the room is one answer too many."""
        latest = self.bus.replay(self.debug.tracks_event)
        if not latest:
            return []
        return to_plain(latest[-1][1]).get("tracks", [])

    def _llm(self) -> dict:
        calls = [call for tap in self.taps.values() for call in tap.calls]
        calls.sort(key=lambda call: call["at"], reverse=True)
        shown = calls[: self.debug.keep_samples]

        return {
            "agents": sorted(self.taps),
            "input_tokens": sum(c["input_tokens"] for c in calls),
            "output_tokens": sum(c["output_tokens"] for c in calls),
            "cost": round(sum(self._cost(c) for c in calls), 4),
            "currency": self.debug.currency,
            "calls": shown,
        }

    def _cost(self, call: dict) -> float:
        """Zero when the model is not priced, and the panel says so. A made-up
        cost is worse than an absent one."""
        price = self.debug.cost_per_million.get(call["model"])
        if not price:
            return NOTHING
        per_million = 1_000_000.0
        return (call["input_tokens"] * price[0] + call["output_tokens"] * price[-1]) / per_million

    # --- the only thing it does -------------------------------------------

    def _on_event(self, event: str, payload) -> None:
        now = time.monotonic()
        with self._lock:
            self.counts[event] = self.counts.get(event, 0) + 1

            window = self.rates.get(event)
            if window is not None:
                window.mark(now)

            for name, pair in self.debug.latencies.items():
                gauge = self.latencies[name]
                if event == pair[0]:
                    gauge.begin(now)
                elif event == pair[-1]:
                    gauge.end(now)

            if event not in self.debug.noisy_events:
                self.recent.append({"at": time.time(), "event": event,
                                    "payload": to_plain(payload)})
