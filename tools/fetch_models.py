#!/usr/bin/env python3
"""Download the models a clone does not carry.

They stay out of git because git keeps every version of a binary forever, and
a pull on a school's connection should not cost sixty megabytes of voice. One
command instead:

    python tools/fetch_models.py            # what the robot needs to see and speak
    python tools/fetch_models.py --hindi    # and a Hindi voice
    python tools/fetch_models.py --check    # say what is missing, download nothing

Anything already present at the right size is left alone, so running it twice
costs nothing.
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"

CHUNK = 1 << 16
MEGABYTE = 1024 * 1024
TIMEOUT_SECONDS = 60

ZOO = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models"
VOICES = "https://huggingface.co/rhasspy/piper-voices/resolve/main"


@dataclass(frozen=True)
class Model:
    path: str
    url: str
    # A floor, not an exact size. It catches the one failure that matters:
    # a Git LFS pointer or an HTML error page saved under a model's name,
    # which loads as garbage and fails somewhere far from here.
    min_bytes: int
    why: str


CORE = [
    Model("face_detection_yunet_2023mar.onnx",
          f"{ZOO}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
          200_000, "finding faces"),
    Model("face_recognition_sface_2021dec.onnx",
          f"{ZOO}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
          30_000_000, "putting names on faces"),
    Model("piper/en_US-lessac-medium.onnx",
          f"{VOICES}/en/en_US/lessac/medium/en_US-lessac-medium.onnx",
          50_000_000, "an English voice"),
    Model("piper/en_US-lessac-medium.onnx.json",
          f"{VOICES}/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json",
          1_000, "that voice's settings"),
]

HINDI = [
    Model("piper/hi_IN-pratham-medium.onnx",
          f"{VOICES}/hi/hi_IN/pratham/medium/hi_IN-pratham-medium.onnx",
          50_000_000, "a Hindi voice"),
    Model("piper/hi_IN-pratham-medium.onnx.json",
          f"{VOICES}/hi/hi_IN/pratham/medium/hi_IN-pratham-medium.onnx.json",
          1_000, "that voice's settings"),
]


def present(model: Model) -> bool:
    target = MODELS / model.path
    return target.exists() and target.stat().st_size >= model.min_bytes


def fetch(model: Model) -> bool:
    target = MODELS / model.path
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")

    print(f"  downloading {model.path} ({model.why})")
    try:
        with urllib.request.urlopen(model.url, timeout=TIMEOUT_SECONDS) as response, \
                partial.open("wb") as out:
            total = int(response.headers.get("Content-Length") or 0)
            done = 0
            while chunk := response.read(CHUNK):
                out.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r    {done / MEGABYTE:6.1f} / {total / MEGABYTE:.1f} MB", end="", flush=True)
        print()
    except OSError as exc:
        partial.unlink(missing_ok=True)
        print(f"    failed: {exc}")
        return False

    if partial.stat().st_size < model.min_bytes:
        size = partial.stat().st_size
        partial.unlink()
        print(f"    refused: only {size} bytes arrived, which is a pointer or an error page")
        return False

    # Renamed only once it is whole, so an interrupted download never leaves
    # a file that looks like a model and loads as nothing.
    partial.replace(target)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="download the models a clone does not carry")
    parser.add_argument("--hindi", action="store_true", help="also fetch a Hindi voice")
    parser.add_argument("--check", action="store_true", help="report only, download nothing")
    args = parser.parse_args()

    wanted = CORE + (HINDI if args.hindi else [])
    missing = [m for m in wanted if not present(m)]

    for model in wanted:
        state = "missing" if model in missing else "ok"
        print(f"  {state:8} {model.path:42} {model.why}")

    if args.check or not missing:
        print("\n  nothing to download." if not missing else "")
        return 1 if missing else 0

    print()
    failed = [m for m in missing if not fetch(m)]
    if failed:
        print(f"\n  {len(failed)} could not be downloaded; the robot runs without them.")
        return 1

    print("\n  done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
