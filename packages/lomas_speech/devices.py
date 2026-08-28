from __future__ import annotations

from dataclasses import dataclass

from lomas_core.errors import LomasError
from lomas_core.schema import AudioConfig


@dataclass(frozen=True, slots=True)
class AudioInput:
    id: str
    device: str
    zone: str


class InputSet:
    """Microphones as a list with zones, even though there is one today.

    This is the seam for per-desk microphones: thirty desks becomes thirty
    config entries, and the code that routes a question to a zone already
    exists.
    """

    def __init__(self, cfg: AudioConfig) -> None:
        self.cfg = cfg
        self.inputs = [AudioInput(id=i.id, device=i.device, zone=i.zone) for i in cfg.inputs]
        if not self.inputs:
            raise LomasError("speech.audio.inputs must list at least one microphone")

    @property
    def primary(self) -> AudioInput:
        return self.inputs[0]

    def zones(self) -> list[str]:
        return sorted({i.zone for i in self.inputs})

    def by_zone(self, zone: str) -> list[AudioInput]:
        return [i for i in self.inputs if i.zone == zone]

    def by_id(self, input_id: str) -> AudioInput:
        for candidate in self.inputs:
            if candidate.id == input_id:
                return candidate
        known = ", ".join(i.id for i in self.inputs)
        raise LomasError(f"no audio input '{input_id}'. Known: {known}")

    def __len__(self) -> int:
        return len(self.inputs)
