from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol, runtime_checkable

from lomas_core import logging as log
from lomas_core.registry import Registry
from lomas_core.schema import VolumeConfig

AUTO = "auto"
NONE = "none"
SOFTWARE = "software"
ALSA = "alsa"

FULL = 1.0
SILENCE = 0.0
PERCENT = 100
ROUNDING = 3


@runtime_checkable
class VolumeControl(Protocol):
    """One knob, however the hardware happens to provide it."""

    def set(self, level: float) -> float: ...

    def mute(self, muted: bool) -> bool: ...

    @property
    def gain(self) -> float: ...

    def describe(self) -> str: ...


VOLUME_CONTROLS: Registry[VolumeControl] = Registry("volume control")


class Dial:
    """What every control shares: a level, a mute, and a memory of both.

    Subclasses decide what a level *does* - move the card's mixer, or scale
    the samples on the way out - and that is the only difference between
    them.
    """

    name = "dial"

    def __init__(self, cfg: VolumeConfig, device: str = "") -> None:
        self.cfg = cfg
        self.device = device
        self.log = log.get("volume")
        remembered = self._recall()
        self.level = self._within(remembered.get("level", cfg.level))
        self.muted = bool(remembered.get("muted", cfg.muted))
        self.apply()

    # --- the knob ---------------------------------------------------------

    def set(self, level: float) -> float:
        self.level = self._within(level)
        self.apply()
        self._remember()
        return self.level

    def mute(self, muted: bool) -> bool:
        self.muted = bool(muted)
        self.apply()
        self._remember()
        return self.muted

    def nudge(self, steps: float) -> float:
        return self.set(self.level + steps * self.cfg.step)

    @property
    def gain(self) -> float:
        """What the player must multiply samples by. A control that moves
        real hardware leaves the samples alone and returns full."""
        return SILENCE if self.muted else FULL

    def apply(self) -> None:
        """Make the level true of the hardware. Nothing to do by default."""

    @property
    def available(self) -> bool:
        return True

    def describe(self) -> str:
        percent = round(self.level * PERCENT)
        return f"{percent}% {self.name}{' (muted)' if self.muted else ''}"

    def report(self) -> dict:
        return {
            "control": self.name,
            "level": round(self.level, ROUNDING),
            "dial": round(self.level * PERCENT),
            "muted": self.muted,
            "min": self.cfg.min_level,
            "max": self.cfg.max_level,
            "step": self.cfg.step,
            "describe": self.describe(),
        }

    # --- limits and memory -------------------------------------------------

    def _within(self, level: float) -> float:
        return max(self.cfg.min_level, min(self.cfg.max_level, float(level)))

    def _recall(self) -> dict:
        if not self.cfg.remember:
            return {}
        try:
            return json.loads(Path(self.cfg.state_file).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            # No file yet on a new robot, which is the usual case and not
            # worth a line in the log.
            return {}

    def _remember(self) -> None:
        if not self.cfg.remember:
            return
        path = Path(self.cfg.state_file)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"level": self.level, "muted": self.muted}),
                            encoding="utf-8")
        except OSError as exc:
            # A read-only data directory is a robot that forgets its volume,
            # not a robot that stops talking.
            self.log.debug("volume not saved: %s", exc)


def volume_control(cfg: VolumeConfig, device: str = "") -> VolumeControl:
    """The control named in config, or the best one this machine has.

    `auto` prefers the card's own mixer, because that is the knob the
    speaker actually obeys, and falls back to scaling the samples when there
    is no mixer to move - a laptop, a USB dongle, a Pi with an HDMI-only
    card.
    """
    VOLUME_CONTROLS.discover("lomas_speech.volumes")
    if cfg.control != AUTO:
        return VOLUME_CONTROLS.create(cfg.control, cfg, device)

    mixer = VOLUME_CONTROLS.create(ALSA, cfg, device)
    if mixer.available:
        return mixer
    return VOLUME_CONTROLS.create(SOFTWARE, cfg, device)
