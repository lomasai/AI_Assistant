from __future__ import annotations

from lomas_speech.volume import NONE, VOLUME_CONTROLS, Dial

FIXED = "volume fixed"


@VOLUME_CONTROLS.register(NONE)
class NoVolume(Dial):
    """A knob that does nothing, for a robot wired to an amplifier with its
    own dial on the front. The slider moves and the level is remembered; only
    mute reaches the room, because a robot nobody can silence is a robot that
    gets unplugged."""

    name = NONE

    def describe(self) -> str:
        return FIXED
