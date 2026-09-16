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
KILL_GRACE = 0.5
FULL_SCALE = 32768.0


@dataclass(frozen=True, slots=True)
class Endpoint:
    """When a turn is over. `silence_ms` 0 means never early."""

    level: float
    silence_ms: int
    no_speech_seconds: float
    chunk_ms: int


def chunk_peak(pcm: bytes) -> float:
    import array

    samples = array.array("h", pcm[: len(pcm) // SAMPLE_WIDTH * SAMPLE_WIDTH])
    return max((abs(s) for s in samples), default=0) / FULL_SCALE


def read_until_quiet(read, sample_rate: int, seconds: float, endpoint: Endpoint) -> bytes:
    """Raw PCM from `read(n)` until the speaker stops, runs out, or never starts.

    Loudness per chunk, nothing cleverer. The teacher has already decided a
    child is about to speak; all this has to notice is the pause after.
    """
    chunk = max(SAMPLE_WIDTH, int(sample_rate * endpoint.chunk_ms / 1000) * SAMPLE_WIDTH)
    limit = int(seconds * sample_rate) * SAMPLE_WIDTH
    waiting = int(endpoint.no_speech_seconds * sample_rate) * SAMPLE_WIDTH
    enough_quiet = int(endpoint.silence_ms * sample_rate / 1000) * SAMPLE_WIDTH

    captured = bytearray()
    heard = False
    quiet = 0
    while len(captured) < limit:
        block = read(chunk)
        if not block:
            break
        captured += block
        if chunk_peak(block) >= endpoint.level:
            heard, quiet = True, 0
        else:
            quiet += len(block)

        if not enough_quiet:
            continue
        if heard and quiet >= enough_quiet:
            break
        if not heard and len(captured) >= waiting:
            break
    return bytes(captured[:limit])


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

    @property
    def available(self) -> bool:
        return self.backend != NONE

    def describe(self) -> str:
        return self.backend

    def record(self, seconds: float, sample_rate: int, endpoint: Endpoint | None = None) -> bytes:
        """Blocks for at most `seconds` and returns a WAV. Empty if nothing
        captured - a silent room is not an error."""
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
            pcm = read_until_quiet(process.stdout.read, sample_rate, seconds, endpoint)
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
