#!/usr/bin/env python3
"""Is it the speaker, the microphone, or the config?

Audio on a Pi is the classic afternoon-long time sink, and almost all of it
is spent not knowing which of three things is wrong. This plays a tone and
records a clip through exactly the objects the robot uses, and measures the
result rather than asking you whether you heard anything.

    python tools/audio_check.py
    python tools/audio_check.py --mode pi --seconds 5
"""
from __future__ import annotations

import argparse
import math
import struct
import subprocess
import sys
import wave
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "packages"))
sys.path.insert(0, str(ROOT))

from lomas_core.config import load  # noqa: E402
from lomas_core.errors import LomasError  # noqa: E402
from lomas_core.secrets import SECRETS_FILE, load_secrets  # noqa: E402
from lomas_speech.player import Player  # noqa: E402
from lomas_speech.recorder import Recorder  # noqa: E402

TONE_HZ = 440
TONE_SECONDS = 1.0
AMPLITUDE = 9000
FULL_SCALE = 32768.0

# Below this a recording is silence: a muted input, the wrong device, or a
# microphone that is not plugged in where you think it is.
HEARD_SOMETHING = 0.01


def devices(command: str) -> str:
    try:
        done = subprocess.run([command, "-l"], capture_output=True, text=True, timeout=5)
        return done.stdout.strip() or done.stderr.strip() or "(nothing listed)"
    except (OSError, subprocess.SubprocessError):
        return f"({command} is not on PATH)"


def tone(rate: int) -> bytes:
    frames = int(TONE_SECONDS * rate)
    return b"".join(
        struct.pack("<h", int(AMPLITUDE * math.sin(2 * math.pi * TONE_HZ * i / rate)))
        for i in range(frames)
    )


def loudness(wav: bytes) -> tuple[float, float]:
    """Peak and RMS, as a share of full scale. Numbers, because 'did you hear
    it' is the question this tool exists to stop anyone having to answer."""
    with wave.open(BytesIO(wav), "rb") as clip:
        raw = clip.readframes(clip.getnframes())
    if not raw:
        return 0.0, 0.0

    samples = struct.unpack(f"<{len(raw) // 2}h", raw[: len(raw) // 2 * 2])
    peak = max(abs(s) for s in samples) / FULL_SCALE
    rms = math.sqrt(sum(s * s for s in samples) / len(samples)) / FULL_SCALE
    return peak, rms


def main() -> int:
    parser = argparse.ArgumentParser(description="check the robot's ears and voice")
    parser.add_argument("--mode", default="pi")
    parser.add_argument("--config-dir", default=str(ROOT / "config"))
    parser.add_argument("--seconds", type=float, default=0.0)
    parser.add_argument("--skip-play", action="store_true")
    args = parser.parse_args()

    load_secrets(Path(args.config_dir) / SECRETS_FILE)
    cfg = load(args.config_dir, args.mode)
    audio = cfg.speech.audio

    print("=== devices the operating system can see ===")
    print("-- playback --\n" + devices("aplay"))
    print("-- capture --\n" + devices("arecord"))

    print("\n=== what the config asks for ===")
    print(f"  recorder      {audio.recorder}   device={audio.device or '(default)'}")
    print(f"  sample rate   {audio.sample_rate}")
    print(f"  tts engine    {cfg.speech.tts.engine}   player={cfg.speech.tts.player}"
          f"   device={cfg.speech.tts.player_device or '(default)'}")

    player = Player(cfg.speech.tts.player, cfg.speech.tts.player_command,
                    cfg.speech.tts.player_device)
    recorder = Recorder(audio.recorder, audio.device, audio.recorder_command)

    print("\n=== what the robot actually resolved to ===")
    print(f"  out  {player.describe()}")
    print(f"  in   {recorder.describe()}")

    if not args.skip_play:
        print(f"\n=== playing a {TONE_HZ} Hz tone for {TONE_SECONDS:g}s ===")
        if not player.available:
            print("  no player. On Raspberry Pi OS aplay should already be there.")
        else:
            try:
                player.play_pcm(tone(audio.sample_rate), audio.sample_rate)
                print("  sent to the device. If you heard nothing, the robot is not the")
                print("  problem - check alsamixer for a muted or zeroed output, and")
                print("  that the jack is selected rather than HDMI.")
            except LomasError as exc:
                print(f"  failed: {exc}")

    seconds = args.seconds or audio.record_seconds
    print(f"\n=== recording {seconds:g}s - say something ===")
    if not recorder.available:
        print("  no recorder. arecord ships with Raspberry Pi OS; check the PATH,")
        print("  or set speech.audio.recorder.")
        return 1

    try:
        captured = recorder.record(seconds, audio.sample_rate)
    except LomasError as exc:
        print(f"  failed: {exc}")
        return 1

    if not captured:
        print("  nothing was captured at all.")
        return 1

    peak, rms = loudness(captured)
    print(f"  {len(captured)} bytes   peak {peak:.3f}   rms {rms:.4f}")

    if peak < HEARD_SOMETHING:
        print("\n  SILENT. The recording worked but carried no sound, which is")
        print("  almost always one of three things:")
        print("    * the wrong device - run `arecord -l` and set")
        print("      LOMAS__speech__audio__device=plughw:<card>,0")
        print("    * the capture level muted or at zero - `alsamixer`, F4 for")
        print("      capture, then arrow up and press M to unmute")
        print("    * a microphone that is not the one you think it is")
        return 1

    print("\n  HEARD YOU. The microphone works and the robot can be spoken to.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
