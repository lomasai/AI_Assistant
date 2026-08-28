from __future__ import annotations

import os
import time

from lomas_core.errors import LomasError
from lomas_core.schema import WakeConfig
from lomas_speech.types import WakeEvent
from lomas_speech.wake import WAKE_WORDS

CERTAIN = 1.0
NO_DETECTION = -1


@WAKE_WORDS.register("porcupine")
class PorcupineWake:
    """Lighter and more accurate than openWakeWord, and easier to train a
    custom phrase on, at the cost of needing a free Picovoice key."""

    def __init__(self, cfg: WakeConfig) -> None:
        self.cfg = cfg
        self._handle = None
        self._stream = None

    def start(self) -> None:
        try:
            import pvporcupine
        except ImportError as exc:
            raise LomasError(
                "pvporcupine is not installed. pip install pvporcupine, or use "
                "speech.wake.engine: openwakeword."
            ) from exc

        key = os.environ.get(self.cfg.access_key_env, "")
        if not key:
            raise LomasError(
                f"{self.cfg.access_key_env} is not set. Porcupine needs a free "
                "Picovoice access key."
            )
        self._handle = pvporcupine.create(
            access_key=key,
            keyword_paths=[self.cfg.model_path],
            sensitivities=[self.cfg.sensitivity],
        )

    def attach(self, stream) -> None:
        self._stream = stream

    def poll(self) -> WakeEvent | None:
        if self._handle is None or self._stream is None:
            return None
        chunk = self._stream.read()
        if chunk is None or self._handle.process(chunk) == NO_DETECTION:
            return None
        return WakeEvent(
            phrase=self.cfg.phrase, confidence=CERTAIN, zone=self.cfg.zone,
            at=time.monotonic(),
        )

    def stop(self) -> None:
        if self._handle is not None:
            self._handle.delete()
            self._handle = None
        self._stream = None
