from __future__ import annotations

import re
import shutil
import subprocess

from lomas_speech.volume import ALSA, PERCENT, VOLUME_CONTROLS, Dial

AMIXER = "amixer"
CARD_FLAG = "-c"
CARD_IN_DEVICE = re.compile(r"CARD=([^,]+)")
SCONTROL = re.compile(r"Simple mixer control '([^']+)'")
LEVEL_IN_REPLY = re.compile(r"\[(\d+)%\]")
MUTE_ON = "unmute"
MUTE_OFF = "mute"
NO_MIXER = "no mixer"


@VOLUME_CONTROLS.register(ALSA)
class AlsaVolume(Dial):
    """The card's own knob, moved with amixer.

    This is the control a headphone jack or a USB speaker actually obeys,
    and the one that survives whatever is playing: aplay, ffplay, a browser.
    Which knob it is differs per card - the Pi's jack calls it PCM, a DAC
    calls it Digital, a USB speaker calls it Speaker - so the mixer is found
    rather than assumed.
    """

    name = ALSA

    def __init__(self, cfg, device: str = "") -> None:
        self.card = cfg.card or _card_of(device)
        self.mixer = ""
        super().__init__(cfg, device)

    @property
    def available(self) -> bool:
        if not shutil.which(AMIXER):
            return False
        return bool(self._found_mixer())

    def apply(self) -> None:
        mixer = self._found_mixer()
        if not mixer:
            return
        percent = round(self.level * PERCENT)
        self._amixer("sset", mixer, f"{percent}%", MUTE_OFF if self.muted else MUTE_ON)

    def describe(self) -> str:
        mixer = self._found_mixer()
        where = f"{mixer} on {self.card}" if self.card else mixer
        return f"{super().describe()} via {where}" if mixer else NO_MIXER

    # --- amixer -----------------------------------------------------------

    def _found_mixer(self) -> str:
        """The first configured knob this card actually has. Cached: it is
        two processes per lookup and the card does not grow new controls."""
        if self.mixer:
            return self.mixer
        reply = self._amixer("scontrols")
        have = set(SCONTROL.findall(reply))
        for wanted in self.cfg.mixers:
            if wanted in have:
                self.mixer = wanted
                break
        return self.mixer

    def _amixer(self, *args: str) -> str:
        argv = [AMIXER]
        if self.card:
            argv += [CARD_FLAG, self.card]
        try:
            done = subprocess.run([*argv, *args], capture_output=True, text=True,
                                  timeout=self.cfg.mixer_timeout_seconds, check=False)
        except (OSError, subprocess.SubprocessError) as exc:
            # A mixer that cannot be moved must not stop a lesson; the robot
            # keeps its current loudness and says why once.
            self.log.debug("amixer failed: %s", exc)
            return ""
        return done.stdout

    def reads(self) -> float:
        """What the card says it is set to, which is not always what was
        asked for: some cards quantise to a handful of steps."""
        mixer = self._found_mixer()
        if not mixer:
            return self.level
        found = LEVEL_IN_REPLY.search(self._amixer("sget", mixer))
        return int(found.group(1)) / PERCENT if found else self.level


def _card_of(device: str) -> str:
    """The card name out of an ALSA device string.

    `plughw:CARD=Headphones,DEV=0` is already in config because that is
    where the speaker is, so the mixer does not need configuring twice.
    """
    found = CARD_IN_DEVICE.search(device or "")
    return found.group(1) if found else ""
