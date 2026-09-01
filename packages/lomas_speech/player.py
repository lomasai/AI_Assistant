from __future__ import annotations

import io
import shlex
import shutil
import subprocess
import sys
import threading
import wave
from pathlib import Path

from lomas_core import logging as log
from lomas_core.errors import LomasError

# No dependency on purpose. Raspberry Pi OS has aplay, macOS has afplay, and
# Windows has winsound in the standard library. A teaching robot that needs a
# pip install before it can make a sound is a robot that arrives mute.

WINDOWS = "win32"
DARWIN = "darwin"

AUTO = "auto"
NONE = "none"
WINSOUND = "winsound"

WAV = ".wav"
MP3 = ".mp3"

# Ordered by how likely each is to be present on the machine it runs on.
UNIX_PLAYERS = {
    "aplay": ["aplay", "-q"],
    "afplay": ["afplay"],
    "paplay": ["paplay"],
    "ffplay": ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"],
    "mpg123": ["mpg123", "-q"],
}

# Only these can handle an mp3, which is what the cloud voice returns.
MP3_PLAYERS = ("ffplay", "mpg123", "afplay")

# How each one is told which card to use. afplay and winsound have no such
# flag; they follow the system default and that is all they offer.
DEVICE_FLAG = {"aplay": "-D", "paplay": "-d", "mpg123": "-a"}

SAMPLE_WIDTH = 2  # piper emits signed 16-bit
MAX_CLIP_SECONDS = 30.0
# How long past the end of a clip a player may take before it is stuck.
STALL_GRACE = 5.0
MONO = 1


def wrap_pcm(raw: bytes, sample_rate: int) -> bytes:
    """Raw PCM into a WAV, in memory.

    Piper writes headerless samples to stdout. Every player on every platform
    wants a header, so this is the one line between a working robot and a
    silent one.
    """
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(MONO)
        out.setsampwidth(SAMPLE_WIDTH)
        out.setframerate(sample_rate)
        out.writeframes(raw)
    return buffer.getvalue()


class Player:
    """Sends finished audio to a speaker, and stops it mid-word when asked.

    `stop` has to return immediately: the teacher's pause button depends on
    it, and a robot that finishes its sentence after being paused is a robot
    that gets switched off.
    """

    def __init__(self, choice: str = AUTO, command: str = "", device: str = "") -> None:
        self.log = log.get("audio")
        self.command = command
        self.device = device
        self.backend = self._choose(choice)
        self._process: subprocess.Popen | None = None
        self._stopped = threading.Event()
        self._lock = threading.RLock()

    @property
    def available(self) -> bool:
        return self.backend != NONE

    def describe(self) -> str:
        return f"{self.backend}:{self.device}" if self.device else self.backend

    def play_pcm(self, raw: bytes, sample_rate: int) -> None:
        if raw:
            self.play_bytes(wrap_pcm(raw, sample_rate), WAV)

    def play_bytes(self, body: bytes, suffix: str) -> None:
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as scratch:
            scratch.write(body)
            path = Path(scratch.name)
        try:
            self.play_file(path)
        finally:
            path.unlink(missing_ok=True)

    def play_file(self, path: Path) -> None:
        """Blocks until the sound has finished or `stop` cuts it off."""
        if self.backend == NONE:
            return
        self._stopped.clear()
        if self.backend == WINSOUND:
            self._play_winsound(path)
            return
        self._play_command(path)

    def stop(self) -> None:
        self._stopped.set()
        with self._lock:
            if self.backend == WINSOUND:
                self._purge_winsound()
                return
            if self._process is not None and self._process.poll() is None:
                self._process.kill()
            self._process = None

    # --- backends ---------------------------------------------------------

    def _choose(self, choice: str) -> str:
        if choice == NONE:
            return NONE
        if choice != AUTO:
            if choice == WINSOUND:
                return WINSOUND if sys.platform == WINDOWS else NONE
            return choice if shutil.which(choice) else NONE

        if self.command:
            return shlex.split(self.command)[0]
        if sys.platform == WINDOWS:
            return WINSOUND
        for name in UNIX_PLAYERS:
            if shutil.which(name):
                return name

        self.log.warning("no audio player found; the robot will be silent")
        return NONE

    def _argv(self, path: Path) -> list[str]:
        if self.command:
            return [*shlex.split(self.command), str(path)]

        argv = [*UNIX_PLAYERS.get(self.backend, [self.backend])]
        if self.device and self.backend in DEVICE_FLAG:
            argv += [DEVICE_FLAG[self.backend], self.device]
        return [*argv, str(path)]

    def _play_command(self, path: Path) -> None:
        if path.suffix == MP3 and self.backend not in MP3_PLAYERS:
            raise LomasError(
                f"'{self.backend}' cannot play mp3. Install ffplay or mpg123, "
                "or use speech.tts.engine: piper."
            )
        try:
            with self._lock:
                self._process = subprocess.Popen(
                    self._argv(path), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
                )
            # A timeout, because some devices accept the open and then never
            # return - an i2s card with nothing clocked on the far end will
            # block aplay forever, and a robot whose voice hangs is a robot
            # whose lesson hangs.
            waited = _seconds(path) + STALL_GRACE
            try:
                _, complaint = self._process.communicate(timeout=waited)
                code = self._process.returncode
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.communicate()
                raise LomasError(
                    f"{self.backend} did not finish within {waited:.0f}s on "
                    f"'{self.device or 'default'}'. The card accepted the audio "
                    "and then stalled; try another with --sweep."
                ) from None
        except OSError as exc:
            raise LomasError(f"cannot run the audio player '{self.backend}': {exc}") from exc
        finally:
            with self._lock:
                self._process = None

        # Swallowing this is how a robot ends up silently miming: aplay exits
        # non-zero for a busy or invalid device and says exactly why.
        if code and not self._stopped.is_set():
            raise LomasError(
                f"{self.backend} failed ({code}): "
                f"{complaint.decode(errors='ignore').strip()[:200] or 'no message'}"
            )

    def _play_winsound(self, path: Path) -> None:
        """Asynchronous, then waited on here.

        Synchronous PlaySound cannot be interrupted: SND_PURGE from another
        thread does not reach it, and the robot finishes its sentence after
        the teacher has pressed pause. Playing async and waiting on our own
        event makes `stop` immediate, which is what that button needs.
        """
        import winsound

        winsound.PlaySound(str(path), winsound.SND_FILENAME | winsound.SND_ASYNC)
        self._stopped.wait(timeout=_seconds(path))
        if not self._stopped.is_set():
            return
        self._purge_winsound()

    def _purge_winsound(self) -> None:
        try:
            import winsound

            winsound.PlaySound(None, winsound.SND_PURGE)
        except (ImportError, RuntimeError):
            pass


def _seconds(path: Path) -> float:
    """How long the clip runs, so the wait ends by itself when nobody stops
    it. An unreadable file falls back to a cap rather than blocking forever."""
    try:
        with wave.open(str(path), "rb") as clip:
            return clip.getnframes() / float(clip.getframerate() or 1)
    except (OSError, wave.Error, ZeroDivisionError):
        return MAX_CLIP_SECONDS
