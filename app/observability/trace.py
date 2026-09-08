from __future__ import annotations

import json
import queue
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from lomas_core import logging as log
from lomas_core.clock import Clock
from lomas_core.events import EventBus, to_plain
from lomas_core.schema import Config

from app.observability.host import Host

ALL_EVENTS = "*"

# One line per record, so a half-finished file is still readable. A run that
# was killed with Ctrl-C is exactly the run you most want to look at.
EVENT = "event"
SAMPLE = "sample"
SPAN = "span"
META = "meta"

STOP = None
FLUSH_EVERY = 50


class Trace:
    """Writes a timeline of a run to one file, for reading somewhere else.

    Deliberately raw. It records what happened and when, and computes almost
    nothing: latencies, rates and blame are all derivable afterwards, and a
    recorder that decides in advance which numbers matter is a recorder that
    hides the one you needed.

    Writing happens on its own thread behind a bounded queue. If the disk
    cannot keep up it drops records and counts them, because a tool for
    finding slowness must never be the slowness.
    """

    def __init__(self, cfg: Config, bus: EventBus, clock: Clock, path: Path) -> None:
        self.cfg = cfg
        self.settings = cfg.trace
        self.bus = bus
        self.clock = clock
        self.path = path
        self.log = log.get("trace")
        self.host = Host()

        self.started = time.time()
        self.written = 0
        self.dropped = 0
        self.counts: dict[str, int] = {}
        self._seen: dict[str, int] = {}

        self._queue: queue.Queue = queue.Queue(maxsize=self.settings.queue_size)
        self._stop = threading.Event()
        self._writer: threading.Thread | None = None
        self._sampler: threading.Thread | None = None

    # --- lifecycle --------------------------------------------------------

    def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = threading.Thread(target=self._write_loop, name="trace", daemon=True)
        self._writer.start()

        self._record(META, {
            "phase": "start",
            "profile": self.cfg.runtime.profile,
            "mode": self.cfg.runtime.mode,
            "at": self.started,
            "config": self._interesting_config(),
        })

        self.bus.subscribe(ALL_EVENTS, self._on_event)
        if self.settings.sample_seconds > 0:
            self._sampler = threading.Thread(target=self._sample_loop, name="trace-host",
                                             daemon=True)
            self._sampler.start()

        self.log.info("recording to %s", self.path)

    def stop(self) -> None:
        self._stop.set()
        if self._sampler is not None:
            self._sampler.join(timeout=self.settings.sample_seconds * 2)
            self._sampler = None

        self._record(META, {"phase": "stop", "at": time.time(),
                            "written": self.written, "dropped": self.dropped,
                            "counts": dict(self.counts)})
        self._queue.put(STOP)
        if self._writer is not None:
            self._writer.join(timeout=self.settings.shutdown_seconds)
            self._writer = None

        self._write_summary()
        self.log.info("wrote %d records to %s", self.written, self.path.name)

    # --- what goes in -----------------------------------------------------

    def span(self, name: str, seconds: float, detail: dict | None = None) -> None:
        """A measured stretch that is not an event. Used sparingly - most
        durations are already the gap between two events."""
        self._record(SPAN, {"name": name, "ms": round(seconds * 1000, 2), **(detail or {})})

    def _on_event(self, event: str, payload: Any) -> None:
        self.counts[event] = self.counts.get(event, 0) + 1

        keep = self.settings.sample_every.get(event, 1)
        if keep > 1:
            # Vision publishes eight times a second. Every line of that would
            # bury the ones that explain a slow answer.
            self._seen[event] = self._seen.get(event, 0) + 1
            if self._seen[event] % keep:
                return

        body = {"event": event}
        if event not in self.settings.payload_exclude:
            body["payload"] = to_plain(payload)
        self._record(EVENT, body)

    def _sample_loop(self) -> None:
        while not self._stop.wait(self.settings.sample_seconds):
            reading = {
                "cores": self.host.cpu(),
                "load": self.host.load(),
                "memory": self.host.memory(),
                "temperature_c": self.host.temperature(),
                "throttled": self.host.throttled(),
            }
            if self.settings.processes:
                reading["processes"] = self.host.processes(self.settings.top_processes)
            self._record(SAMPLE, reading)

    def _record(self, kind: str, body: dict) -> None:
        line = {"t": round(time.time() - self.started, 4), "kind": kind, **body}
        try:
            self._queue.put_nowait(line)
        except queue.Full:
            # Counted, not blocked. The alternative is a profiler that slows
            # the thing it is profiling, which teaches you nothing true.
            self.dropped += 1

    # --- the writer -------------------------------------------------------

    def _write_loop(self) -> None:
        pending = 0
        with self.path.open("w", encoding="utf-8") as out:
            while True:
                line = self._queue.get()
                if line is STOP:
                    out.flush()
                    return
                try:
                    out.write(json.dumps(line, default=str) + "\n")
                    self.written += 1
                    pending += 1
                except (OSError, ValueError) as exc:
                    self.log.error("could not write a trace line: %s", exc)

                if pending >= FLUSH_EVERY:
                    out.flush()
                    pending = 0

    # --- the part you read first -----------------------------------------

    def _write_summary(self) -> None:
        """A short markdown file beside the trace.

        The jsonl is for me; this is so anyone can glance at a run and see
        whether it was slow, without any tooling at all.
        """
        summary = self.path.with_suffix(".md")
        seconds = max(time.time() - self.started, 1e-9)
        busiest = sorted(self.counts.items(), key=lambda kv: -kv[1])[:15]

        lines = [
            f"# {self.path.stem}",
            "",
            f"- profile: `{self.cfg.runtime.profile}` (mode `{self.cfg.runtime.mode}`)",
            f"- ran for: {seconds:.1f}s",
            f"- records: {self.written}" + (f" ({self.dropped} dropped)" if self.dropped else ""),
            f"- events: {sum(self.counts.values())}",
            "",
            "## Busiest events",
            "",
            "| event | count | per second |",
            "| --- | ---: | ---: |",
        ]
        lines += [f"| `{name}` | {count} | {count / seconds:.2f} |" for name, count in busiest]

        memory = self.host.memory()
        if memory:
            lines += [
                "",
                "## Machine at the end",
                "",
                f"- this process: {memory.get('process_mb', 0)} MB",
                f"- free: {memory.get('available_mb', 0)} of {memory.get('total_mb', 0)} MB",
                f"- temperature: {self.host.temperature()} C",
                f"- throttled: {', '.join(self.host.throttled()) or 'no'}",
            ]

        lines += ["", "Push this folder and the jsonl beside it for analysis.", ""]
        summary.write_text("\n".join(lines), encoding="utf-8")

    def _interesting_config(self) -> dict:
        """The settings that change how fast it runs. Enough to tell two runs
        apart without shipping the whole config, which carries no secrets but
        is long enough to bury the six numbers that matter."""
        cfg = self.cfg
        source = cfg.sources[0] if cfg.sources else None
        return {
            "detect_fps": cfg.face.detect_fps,
            "downscale_width": cfg.face.downscale_width,
            "detector": cfg.face.detector,
            "embedder": cfg.face.embedder,
            "recognition": cfg.privacy.recognition_enabled,
            "camera": f"{source.kind} {source.width}x{source.height}@{source.fps}" if source else "",
            "mjpeg_fps": cfg.web.mjpeg_fps,
            "surfaces": cfg.web.surfaces,
            "llm": cfg.llm.provider,
            "tts": cfg.speech.tts.engine,
            "stt": cfg.speech.stt.engine,
            "hardware": cfg.hardware.backend if cfg.hardware.enabled else "off",
        }


def trace_path(cfg: Config, clock: Clock) -> Path:
    """`2026-09-08_1435_pi_a3f2.jsonl` - sortable, and says which run it was
    without opening it."""
    from lomas_store.repos.base import new_id

    stamp = datetime.now().strftime(cfg.trace.name_format)
    label = cfg.runtime.profile or cfg.runtime.mode
    return Path(cfg.trace.directory) / f"{stamp}_{label}_{new_id()[:4]}.jsonl"


def build_trace(cfg: Config, bus: EventBus, clock: Clock) -> Trace | None:
    if not cfg.trace.enabled:
        return None
    return Trace(cfg, bus, clock, trace_path(cfg, clock))
