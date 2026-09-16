from __future__ import annotations

import queue
import threading
from pathlib import Path

from lomas_core import logging as log
from lomas_core.errors import LomasError
from lomas_core.schema import TtsConfig
from lomas_speech.player import Player
from lomas_speech.tts import TTS_ENGINES
from lomas_speech.types import SpeechHandle

SILENT = 0.0
TALKING = 0.6
DONE = None
WARMUP_TEXT = "Hello."


@TTS_ENGINES.register("piper_python")
class PiperPythonTts:
    """piper loaded once, inside the robot, and spoken a sentence at a time.

    The `piper` engine starts a process per utterance. The Pi's second trace
    showed what that costs: six seconds loading the voice on one core, seven
    more synthesising the whole paragraph, and only then a sound - fourteen
    seconds of silence before every paragraph. Here the voice loads at boot,
    and each sentence plays as soon as it exists while the next one is made.

    Needs `pip install piper-tts`, which the `piper` command came from anyway.
    """

    def __init__(self, cfg: TtsConfig) -> None:
        self.cfg = cfg
        self.player = Player(cfg.player, cfg.player_command, cfg.player_device)
        self.log = log.get("tts")
        self._voices: dict[Path, object] = {}
        self._handle: SpeechHandle | None = None
        self._loading = threading.Lock()
        self._lock = threading.RLock()
        if cfg.preload:
            threading.Thread(target=self._preload, name="piper-preload", daemon=True).start()

    def _preload(self) -> None:
        # At boot, not on the first sentence: the greeting is the one a class
        # notices waiting for. A failure here is said again on first use.
        try:
            voice = self._voice(self._model_for(self.cfg.fallback_language))
            for _chunk in voice.synthesize(WARMUP_TEXT):
                pass
        except LomasError as exc:
            self.log.debug("voice not preloaded: %s", exc)

    def _model_for(self, language: str) -> Path:
        name = self.cfg.voice.get(language) or self.cfg.voice.get(self.cfg.fallback_language)
        if not name:
            raise LomasError(
                f"no piper voice configured for '{language}'. Add it under speech.tts.voice."
            )
        return Path(self.cfg.model_dir) / f"{name}.onnx"

    def _voice(self, model: Path):
        with self._loading:
            if model in self._voices:
                return self._voices[model]
            try:
                from piper import PiperVoice
            except ImportError as exc:
                raise LomasError(
                    "piper-tts is not installed. pip install piper-tts, or use "
                    "speech.tts.engine: piper."
                ) from exc
            if not model.exists():
                raise LomasError(
                    f"piper voice not found at {model}. Run `python tools/fetch_models.py`."
                )
            self._voices[model] = PiperVoice.load(str(model))
            return self._voices[model]

    def speak(self, text: str, language: str = "") -> SpeechHandle:
        language = language or self.cfg.fallback_language
        voice = self._voice(self._model_for(language))

        self.stop()
        handle = SpeechHandle(text=text, language=language)
        with self._lock:
            self._handle = handle

        sentences: queue.Queue = queue.Queue()
        threading.Thread(target=self._synthesise, args=(voice, text, handle, sentences),
                         name="piper-synth", daemon=True).start()
        threading.Thread(target=self._play, args=(handle, sentences),
                         name="piper-play", daemon=True).start()
        return handle

    def _synthesise(self, voice, text: str, handle: SpeechHandle, sentences: queue.Queue) -> None:
        try:
            for chunk in voice.synthesize(text):
                if handle.cancelled:
                    break
                sentences.put((chunk.audio_int16_bytes, chunk.sample_rate))
        except Exception as exc:  # onnxruntime raises its own types
            sentences.put(LomasError(f"piper could not synthesise: {exc}"))
        finally:
            sentences.put(DONE)

    def _play(self, handle: SpeechHandle, sentences: queue.Queue) -> None:
        # Kept apart from synthesis so the next sentence is being made while
        # this one plays; after the first, there is no gap to wait out.
        try:
            while (item := sentences.get()) is not DONE:
                if isinstance(item, LomasError):
                    raise item
                if handle.cancelled:
                    continue
                audio, rate = item
                self.player.play_pcm(audio, rate)
        except LomasError as exc:
            if not handle.cancelled:
                handle.fail(str(exc))
        finally:
            handle.finish()

    def stop(self) -> None:
        with self._lock:
            if self._handle is not None and not self._handle.done:
                self._handle.cancel()
        self.player.stop()

    def amplitude(self) -> float:
        with self._lock:
            speaking = self._handle is not None and not self._handle.done
        return TALKING if speaking else SILENT
