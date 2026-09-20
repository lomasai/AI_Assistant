from __future__ import annotations

import shlex
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

from lomas_core import logging as log
from lomas_core.errors import LomasError

# The mirror of player.py, and for the same reason: Raspberry Pi OS has
# arecord, and a robot that needs a pip install before it can hear is a robot
# that arrives deaf. sounddevice is there for machines that have no arecord.

WINDOWS = "win32"

AUTO = "auto"
NONE = "none"
ARECORD = "arecord"
SOUNDDEVICE = "sounddevice"

SAMPLE_WIDTH = 2
MONO = 1
# The quiet end of a turn is read off a low share of it. The loud end is
# the mean of its loudest few chunks instead: a child speaking for two
# seconds of a fifteen second recording is a tenth of it, and every share
# high enough to be robust was still measuring the room.
QUIET_SHARE = 0.25
LOUDEST_SHARE = 0.05
LOUDEST_MIN = 3
KILL_GRACE = 0.5
FULL_SCALE = 32768.0


@dataclass(frozen=True, slots=True)
class Endpoint:
    """When a turn is over. `silence_ms` 0 means never early.

    Speech is measured against the room, not a fixed number. On the Pi a
    fixed peak of 0.02 sat below the USB microphone's own hiss, so every
    chunk counted as speech and every recording ran its full fifteen seconds.
    """

    silence_ms: int
    no_speech_seconds: float
    chunk_ms: int
    # Speech sits this far from the room's own noise towards the loudest
    # thing heard so far. A share, not a level: the Pi's room measured 0.08
    # where a fixed "definitely a voice" of 0.03 was below the hiss itself,
    # and every chunk counted as speech.
    speech_fraction: float = 0.35
    # There is a pause to find when the loud parts stand this far above the
    # quiet ones, by either measure. Two, because a hot microphone and a
    # quiet one disagree about what a big difference is: the Pi measured
    # 0.08 against 0.20 one evening and 0.033 against 0.044 the next.
    min_gap_rms: float = 0.01
    min_gap_ratio: float = 1.25
    min_rms: float = 0.005  # digital silence is never speech


@dataclass(slots=True)
class Turn:
    """What happened while listening. Traced, so the next run shows the
    room's real numbers instead of leaving them to be guessed."""

    pcm: bytes = b""
    stopped: str = "ended"  # pause | no_speech | limit | ended
    seconds: float = 0.0
    floor_rms: float = 0.0
    # The loud end as the decision saw it, and the single loudest chunk. Both,
    # because a tool that prints one and decides on the other sends everybody
    # looking in the wrong place.
    loud_rms: float = 0.0
    loudest_rms: float = 0.0
    spoke_seconds: float = 0.0

    def summary(self) -> dict:
        return {
            "stopped": self.stopped,
            "seconds": round(self.seconds, 2),
            "floor_rms": round(self.floor_rms, 4),
            "loud_rms": round(self.loud_rms, 4),
            "loudest_rms": round(self.loudest_rms, 4),
            "spoke_seconds": round(self.spoke_seconds, 2),
        }


def chunk_rms(pcm: bytes) -> float:
    import array

    samples = array.array("h", pcm[: len(pcm) // SAMPLE_WIDTH * SAMPLE_WIDTH])
    if not samples:
        return 0.0
    return (sum(s * s for s in samples) / len(samples)) ** 0.5 / FULL_SCALE


def room_floor(levels: list[float]) -> float:
    """The quiet end of what has been heard so far - the lower quartile, not
    the minimum, so one dead buffer does not make every breath "speech"."""
    return at_share(levels, QUIET_SHARE)


def room_loud(levels: list[float]) -> float:
    """The loud end: the mean of the loudest few chunks.

    Not the maximum, so one scraped chair is not the level a child has to
    shout over, and not a percentile either - speech is a small share of a
    turn that is mostly the pause before and after it.
    """
    if not levels:
        return 0.0
    take = max(LOUDEST_MIN, int(len(levels) * LOUDEST_SHARE))
    loudest = sorted(levels, reverse=True)[:take]
    return sum(loudest) / len(loudest)


def at_share(levels: list[float], share: float) -> float:
    ordered = sorted(levels)
    if not ordered:
        return 0.0
    return ordered[min(len(ordered) - 1, int(len(ordered) * share))]


def worth_splitting(floor: float, loud: float, endpoint: Endpoint) -> bool:
    """Whether there is any difference between the loud and quiet parts to
    find a pause in. Without one the turn records to the end: cutting a child
    off is worse than recording air."""
    return (loud - floor) >= endpoint.min_gap_rms or (
        floor > 0 and loud / floor >= endpoint.min_gap_ratio
    )


def read_until_quiet(read, sample_rate: int, seconds: float, endpoint: Endpoint) -> Turn:
    """Raw PCM from `read(n)` until the speaker stops, runs out, or never starts.

    Loudness per chunk against the room, nothing cleverer. The teacher has
    already decided a child is about to speak; all this has to notice is the
    pause after.
    """
    chunk = max(SAMPLE_WIDTH, int(sample_rate * endpoint.chunk_ms / 1000) * SAMPLE_WIDTH)
    limit = int(seconds * sample_rate) * SAMPLE_WIDTH
    waiting = int(endpoint.no_speech_seconds * sample_rate) * SAMPLE_WIDTH
    enough_quiet = int(endpoint.silence_ms * sample_rate / 1000) * SAMPLE_WIDTH
    per_second = sample_rate * SAMPLE_WIDTH

    turn = Turn(stopped="limit")
    captured = bytearray()
    levels: list[float] = []
    heard = False
    quiet = 0
    spoke = 0
    while len(captured) < limit:
        block = read(chunk)
        if not block:
            turn.stopped = "ended"
            break
        captured += block
        level = chunk_rms(block)
        levels.append(level)
        turn.loudest_rms = max(turn.loudest_rms, level)

        floor, loud = room_floor(levels), room_loud(levels)
        speaking = floor + (loud - floor) * endpoint.speech_fraction
        # Three questions in order: is there anything at all, is there enough
        # difference between the loud and quiet parts to find a pause in, and
        # is this chunk one of the loud ones. Uniform sound counts as speech
        # rather than silence: cutting a child off is worse than recording air.
        if level < endpoint.min_rms:
            quiet += len(block)
        elif not worth_splitting(floor, loud, endpoint) or level >= speaking:
            heard, quiet = True, 0
            spoke += len(block)
        else:
            quiet += len(block)

        if not enough_quiet:
            continue
        if heard and quiet >= enough_quiet:
            turn.stopped = "pause"
            break
        if not heard and len(captured) >= waiting:
            turn.stopped = "no_speech"
            break

    turn.pcm = bytes(captured[:limit])
    turn.seconds = len(turn.pcm) / per_second
    turn.floor_rms = room_floor(levels)
    turn.loud_rms = room_loud(levels)
    turn.spoke_seconds = spoke / per_second
    return turn


def loudness(wav: bytes) -> tuple[float, float]:
    """Peak and RMS of a WAV, as a share of full scale.

    Whisper invents words when handed silence - a full stop, or a stray
    "So, let's go." - so knowing how loud a clip is before sending it saves
    both an API call and a sentence nobody said.
    """
    import struct
    import wave
    from io import BytesIO

    try:
        with wave.open(BytesIO(wav), "rb") as clip:
            raw = clip.readframes(clip.getnframes())
    except (wave.Error, EOFError):
        return 0.0, 0.0
    if not raw:
        return 0.0, 0.0

    usable = len(raw) // SAMPLE_WIDTH
    samples = struct.unpack(f"<{usable}h", raw[: usable * SAMPLE_WIDTH])
    peak = max(abs(s) for s in samples) / FULL_SCALE
    rms = (sum(s * s for s in samples) / len(samples)) ** 0.5 / FULL_SCALE
    return peak, rms


class Recorder:
    """Captures a child's turn as WAV bytes.

    The teacher's button decides who speaks and when; with an `Endpoint`,
    arecord stops at the pause after they finish rather than at a fixed
    length. A custom command and sounddevice still record the full length.
    """

    def __init__(self, choice: str = AUTO, device: str = "", command: str = "") -> None:
        self.log = log.get("audio")
        self.device = device
        self.command = command
        self.backend = self._choose(choice)
        self._process: subprocess.Popen | None = None
        self._lock = threading.RLock()
        self.last_turn: Turn | None = None

    @property
    def available(self) -> bool:
        return self.backend != NONE

    def describe(self) -> str:
        return self.backend

    def record(self, seconds: float, sample_rate: int, endpoint: Endpoint | None = None) -> bytes:
        """Blocks for at most `seconds` and returns a WAV. Empty if nothing
        captured - a silent room is not an error."""
        self.last_turn = None
        if self.backend == NONE:
            raise LomasError(
                "no microphone backend. On Raspberry Pi OS arecord is already "
                "there; elsewhere pip install sounddevice, or set "
                "speech.audio.recorder."
            )
        if self.backend == SOUNDDEVICE:
            return self._with_sounddevice(seconds, sample_rate)
        if endpoint is not None and endpoint.silence_ms and self.backend == ARECORD and not self.command:
            return self._streaming(seconds, sample_rate, endpoint)
        return self._with_command(seconds, sample_rate)

    def stop(self) -> None:
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                self._process.kill()
            self._process = None

    # --- backends ---------------------------------------------------------

    def _choose(self, choice: str) -> str:
        if choice == NONE:
            return NONE
        if choice != AUTO:
            if choice == SOUNDDEVICE:
                return SOUNDDEVICE if _has_sounddevice() else NONE
            return choice if shutil.which(choice) else NONE

        if self.command:
            return shlex.split(self.command)[0]
        if sys.platform != WINDOWS and shutil.which(ARECORD):
            return ARECORD
        if _has_sounddevice():
            return SOUNDDEVICE

        self.log.warning("no microphone backend; the robot cannot hear")
        return NONE

    def _argv(self, path: Path, seconds: float, sample_rate: int) -> list[str]:
        if self.command:
            return [*shlex.split(self.command), str(path)]

        argv = [self.backend, "-q", "-f", "S16_LE", "-c", str(MONO),
                "-r", str(sample_rate), "-t", "wav", "-d", str(int(round(seconds)))]
        if self.device:
            argv += ["-D", self.device]
        return [*argv, str(path)]

    def _with_command(self, seconds: float, sample_rate: int) -> bytes:
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as scratch:
            path = Path(scratch.name)
        try:
            with self._lock:
                self._process = subprocess.Popen(
                    self._argv(path, seconds, sample_rate),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
            _, complaint = self._process.communicate(timeout=seconds + KILL_GRACE * 4)
            if self._process.returncode and complaint:
                raise LomasError(
                    f"{self.backend} failed: {complaint.decode(errors='ignore').strip()[:160]}"
                )
            return path.read_bytes() if path.exists() else b""
        except subprocess.TimeoutExpired:
            self.stop()
            return b""
        except OSError as exc:
            raise LomasError(f"cannot run '{self.backend}': {exc}") from exc
        finally:
            with self._lock:
                self._process = None
            path.unlink(missing_ok=True)

    def _streaming(self, seconds: float, sample_rate: int, endpoint: Endpoint) -> bytes:
        from lomas_speech.player import wrap_pcm

        argv = [ARECORD, "-q", "-f", "S16_LE", "-c", str(MONO), "-r", str(sample_rate),
                "-t", "raw", "-d", str(int(seconds) + 1)]
        if self.device:
            argv += ["-D", self.device]
        try:
            with self._lock:
                process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                self._process = process
        except OSError as exc:
            raise LomasError(f"cannot run '{self.backend}': {exc}") from exc

        try:
            turn = read_until_quiet(process.stdout.read, sample_rate, seconds, endpoint)
            self.last_turn = turn
            pcm = turn.pcm
        finally:
            if process.poll() is None:
                process.kill()
            try:
                _, complaint = process.communicate(timeout=KILL_GRACE * 4)
            except subprocess.TimeoutExpired:
                complaint = b""
            with self._lock:
                if self._process is process:
                    self._process = None

        if not pcm and complaint:
            raise LomasError(f"{self.backend} failed: {complaint.decode(errors='ignore').strip()[:160]}")
        return wrap_pcm(pcm, sample_rate) if pcm else b""

    def _with_sounddevice(self, seconds: float, sample_rate: int) -> bytes:
        import sounddevice
        from lomas_speech.player import wrap_pcm

        frames = int(seconds * sample_rate)
        captured = sounddevice.rec(
            frames, samplerate=sample_rate, channels=MONO, dtype="int16",
            device=self.device or None,
        )
        sounddevice.wait()
        return wrap_pcm(captured.tobytes(), sample_rate)


def _has_sounddevice() -> bool:
    try:
        import sounddevice  # noqa: F401
        return True
    except Exception:
        # It raises OSError when portaudio itself is missing, not ImportError.
        return False
