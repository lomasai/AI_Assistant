from __future__ import annotations

import shutil
import subprocess
import sys
import threading
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


class Recorder:
    """Captures a stretch of microphone audio as WAV bytes.

    Fixed-length on purpose. Voice activity detection in a room of forty
    children is a research project; a teacher deciding when a child is
    speaking is a button, and the button is right far more often.
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

    def record(self, seconds: float, sample_rate: int) -> bytes:
        """Blocks for `seconds` and returns a WAV. Empty if nothing captured -
        a silent room is not an error."""
        if self.backend == NONE:
            raise LomasError(
                "no microphone backend. On Raspberry Pi OS arecord is already "
                "there; elsewhere pip install sounddevice, or set "
                "speech.audio.recorder."
            )
        if self.backend == SOUNDDEVICE:
            return self._with_sounddevice(seconds, sample_rate)
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
            return self.command.split()[0]
        if sys.platform != WINDOWS and shutil.which(ARECORD):
            return ARECORD
        if _has_sounddevice():
            return SOUNDDEVICE

        self.log.warning("no microphone backend; the robot cannot hear")
        return NONE

    def _argv(self, path: Path, seconds: float, sample_rate: int) -> list[str]:
        if self.command:
            return [*self.command.split(), str(path)]

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
