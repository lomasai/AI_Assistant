from __future__ import annotations

import time

from lomas_core.errors import LomasError
from lomas_core.schema import WakeConfig
from lomas_speech.types import WakeEvent
from lomas_speech.wake import WAKE_WORDS


@WAKE_WORDS.register("openwakeword")
class OpenWakeWord:
    """The default. No API key, ONNX, roughly a tenth of a core on a Pi 4."""

    def __init__(self, cfg: WakeConfig) -> None:
        self.cfg = cfg
        self._model = None
        self._stream = None

    def start(self) -> None:
        try:
            from openwakeword.model import Model
        except ImportError as exc:
            raise LomasError(
                "openwakeword is not installed. pip install openwakeword, or use "
                "speech.wake.engine: keyboard."
            ) from exc
        self._model = Model(wakeword_models=[self.cfg.model_path])

    def poll(self) -> WakeEvent | None:
        if self._model is None or self._stream is None:
            return None

        chunk = self._stream.read()
        if chunk is None:
            return None

        scores = self._model.predict(chunk)
        best = max(scores.values()) if scores else 0.0
        if best < self.cfg.sensitivity:
            return None
        return WakeEvent(
            phrase=self.cfg.phrase, confidence=float(best), zone=self.cfg.zone,
            at=time.monotonic(),
        )

    def attach(self, stream) -> None:
        """Audio capture is owned elsewhere; this engine only scores chunks."""
        self._stream = stream

    def stop(self) -> None:
        self._model = None
        self._stream = None
