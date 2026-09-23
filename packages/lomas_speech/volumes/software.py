from __future__ import annotations

from lomas_speech.volume import FULL, SILENCE, SOFTWARE, VOLUME_CONTROLS, Dial


@VOLUME_CONTROLS.register(SOFTWARE)
class SoftwareVolume(Dial):
    """Loudness by scaling the samples on the way to the player.

    Works with no mixer, no card and no root: a laptop, a USB dongle, a Pi
    whose only playback device is HDMI. It cannot go above what the card is
    already set to, which is why `auto` prefers the real mixer.
    """

    name = SOFTWARE

    @property
    def gain(self) -> float:
        if self.muted:
            return SILENCE
        return min(FULL, self.level ** self.cfg.curve)
