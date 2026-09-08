#!/usr/bin/env python3
"""What was slow, and what was to blame.

Reads a trace written by a run and answers three questions in order:
how long each thing took, what the machine was doing while it took that
long, and which process was using the machine.

    python tools/trace_report.py                      # the newest trace
    python tools/trace_report.py data/logs/<file>.jsonl
    python tools/trace_report.py --slowest 20
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "data" / "logs"

# Pairs that bracket something a person waits for. The first event starts the
# clock and the second stops it; anything not in here is derivable by hand.
WAITS = {
    "the robot speaking": ("robot.say", "robot.spoke"),
    "answering a question": ("question.asked", "question.answered"),
    "marking an answer": ("quiz.recorded", "quiz.marked"),
    "listening": ("robot.state", "robot.state"),
    "a step": ("step.entered", "step.exited"),
}

BUSY_CORE = 70.0
HOT_C = 70.0
SLOW_MS = 1000.0
PERCENTILES = (0.5, 0.9)


def newest() -> Path | None:
    traces = sorted(LOGS.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    return next((p for p in traces if p.name != "lomas.jsonl"), None)


def read(path: Path) -> list[dict]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            records.append(json.loads(line))
        except ValueError:
            continue  # a run killed mid-write leaves one ragged line
    return records


def gaps(records: list[dict], first: str, second: str) -> list[float]:
    """Milliseconds between each `first` and the `second` that followed it."""
    started: float | None = None
    found: list[float] = []
    for record in records:
        if record.get("kind") != "event":
            continue
        if record["event"] == first and started is None:
            started = record["t"]
        elif record["event"] == second and started is not None:
            found.append((record["t"] - started) * 1000)
            started = None
    return found


def describe(values: list[float]) -> str:
    if not values:
        return "-"
    ordered = sorted(values)
    at = [ordered[min(len(ordered) - 1, int(p * len(ordered)))] for p in PERCENTILES]
    return f"{len(values):4d}   {statistics.mean(values):8.0f} {at[0]:8.0f} {at[1]:8.0f} {max(values):8.0f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="read a trace")
    parser.add_argument("path", nargs="?", default="")
    parser.add_argument("--slowest", type=int, default=10)
    args = parser.parse_args()

    path = Path(args.path) if args.path else newest()
    if path is None or not path.exists():
        print(f"no trace found in {LOGS}. Run with trace.enabled=true first.")
        return 1

    records = read(path)
    meta = next((r for r in records if r.get("phase") == "start"), {})
    closing = next((r for r in records if r.get("phase") == "stop"), {})
    samples = [r for r in records if r.get("kind") == "sample"]
    spans = [r for r in records if r.get("kind") == "span"]
    events = [r for r in records if r.get("kind") == "event"]
    ran = records[-1]["t"] if records else 0.0

    print(f"=== {path.name} ===")
    print(f"  {ran:.0f}s, {len(events)} events, {len(samples)} samples"
          + (f", {closing['dropped']} dropped" if closing.get("dropped") else ""))
    if meta.get("config"):
        for key, value in meta["config"].items():
            print(f"    {key:16} {value}")

    # --- how long things took --------------------------------------------

    print("\n=== how long each thing took, in milliseconds ===")
    print(f"  {'':26} {'n':>4}   {'mean':>8} {'p50':>8} {'p90':>8} {'worst':>8}")
    for label, (first, second) in WAITS.items():
        if first == second:
            continue
        print(f"  {label:26} {describe(gaps(records, first, second))}")

    by_name: dict[str, list[float]] = {}
    for span in spans:
        by_name.setdefault(span.get("name", "?"), []).append(span.get("ms", 0.0))
    for name, values in sorted(by_name.items()):
        print(f"  {'span: ' + name:26} {describe(values)}")

    # --- the slowest individual moments ----------------------------------

    slow = sorted(spans, key=lambda s: -s.get("ms", 0))[: args.slowest]
    if slow:
        print(f"\n=== the {len(slow)} slowest single calls ===")
        for span in slow:
            detail = " ".join(f"{k}={v}" for k, v in span.items()
                              if k not in ("kind", "t", "ms", "name"))
            flag = "  <-- over a second" if span.get("ms", 0) > SLOW_MS else ""
            print(f"  t={span['t']:7.1f}  {span.get('ms', 0):8.0f} ms  {span.get('name')}  {detail}{flag}")

    # --- what the machine was doing --------------------------------------

    if samples:
        print("\n=== the machine ===")
        cores = [max(s["cores"]) for s in samples if s.get("cores")]
        temps = [s["temperature_c"] for s in samples if s.get("temperature_c")]
        rss = [s["memory"].get("process_mb", 0) for s in samples if s.get("memory")]
        throttles = {flag for s in samples for flag in s.get("throttled", [])}

        if cores:
            hot = sum(1 for c in cores if c > BUSY_CORE)
            print(f"  busiest core: mean {statistics.mean(cores):.0f}%  peak {max(cores):.0f}%"
                  f"   ({hot} of {len(cores)} samples over {BUSY_CORE:.0f}%)")
        if temps:
            print(f"  temperature : mean {statistics.mean(temps):.1f} C  peak {max(temps):.1f} C"
                  + ("   <-- throttling range" if max(temps) > HOT_C else ""))
        if rss:
            print(f"  this process: {rss[0]:.0f} MB -> {rss[-1]:.0f} MB (peak {max(rss):.0f})")
        if throttles:
            print(f"  THROTTLED   : {', '.join(sorted(throttles))}")

        # --- and who was using it -----------------------------------------

        totals: dict[str, list[float]] = {}
        for sample in samples:
            for process in sample.get("processes", []):
                totals.setdefault(process["name"], []).append(process["cpu"])

        if totals:
            print("\n=== which process was using the CPU ===")
            print(f"  {'process':22} {'mean %':>8} {'peak %':>8} {'seen':>6}")
            ranked = sorted(totals.items(), key=lambda kv: -statistics.mean(kv[1]))[:12]
            for name, values in ranked:
                print(f"  {name:22} {statistics.mean(values):8.1f} {max(values):8.1f} {len(values):6d}")

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
