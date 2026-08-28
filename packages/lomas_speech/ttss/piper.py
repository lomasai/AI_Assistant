from __future__ import annotations

import shutil
import subprocess
import threading
from pathlib import Path

from lomas_core.errors import LomasError
from lomas_core.schema import TtsConfig
from lomas_speech.tts import TTS_ENGINES
from lomas_speech.types import SpeechHandle

SILENT = 0.0
TALKING = 0.6
GRACE_SECONDS = 0.2


@TTS_ENGINES.register("piper")
class PiperTts:
    """The default. Runs locally, faster than realtime on a Pi 4, and has
    Hindi and other Indic voices - which is what makes the multilingual
    promise affordable."""

    def __init__(self, cfg: TtsConfig) -> None:
        self.cfg = cfg
        self._process: subprocess.Popen | None = None
        self._handle: SpeechHandle | None = None
        self._lock = threading.RLock()

    def _voice_for(self, language: str) -> Path:
        voice = self.cfg.voice.get(language) or self.cfg.voice.get(self.cfg.fallback_language)
        if not voice:
            raise LomasError(
                f"no piper voice configured for '{language}'. Add it under "
                "speech.tts.voice."
            )
        return Path(self.cfg.model_dir) / f"{voice}.onnx"

    def speak(self, text: str, language: str = "") -> SpeechHandle:
        language = language or self.cfg.fallback_language
        binary = shutil.which(self.cfg.binary)
        if binary is None:
            raise LomasError(
                f"'{self.cfg.binary}' is not on PATH. Install piper, or use "
                "speech.tts.engine: null."
            )

        model = self._voice_for(language)
        if not model.exists():
            raise LomasError(f"piper voice not found at {model}")

        self.stop()
        handle = SpeechHandle(text=text, language=language)
        with self._lock:
            self._handle = handle
            self._process = subprocess.Popen(
                [binary, "--model", str(model), "--output-raw"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            )
        threading.Thread(target=self._run, args=(text, handle), daemon=True).start()
        return handle

    def _run(self, text: str, handle: SpeechHandle) -> None:
        try:
            self._process.communicate(text.encode("utf-8"))
        finally:
            handle.finish()

    def stop(self) -> None:
        """Must cut off mid-sentence. The teacher's pause button depends on
        this returning quickly, so the process is killed rather than asked."""
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                self._process.kill()
                self._process.wait(timeout=GRACE_SECONDS)
            if self._handle is not None and not self._handle.done:
                self._handle.cancel()
            self._process = None

    def amplitude(self) -> float:
        with self._lock:
            speaking = self._process is not None and self._process.poll() is None
        return TALKING if speaking else SILENT
