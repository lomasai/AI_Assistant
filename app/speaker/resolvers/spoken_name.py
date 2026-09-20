from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

from app.speaker.resolver import RESOLVERS
from app.speaker.types import Heard, Speaker

PUNCTUATION = ".,!?;:\"'()-—"
WORDS = 2  # the longest name looked for, in words


def tidy(word: str) -> str:
    return word.strip(PUNCTUATION).lower()


def looks_like(said: str, name: str) -> float:
    return SequenceMatcher(None, said, name).ratio()


@RESOLVERS.register("spoken_name")
class SpokenName:
    """The child said their name: "I am Akshay, why do leaves fall?"

    Matched loosely on purpose. Speech to text returns Akshaya, Akash and
    action for the same child, and a robot that only accepts the spelling in
    the register is a robot nobody can talk to. The name is then taken off
    the question, so the tutor answers what was asked and not the
    introduction.
    """

    name = "spoken_name"

    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def resolve(self, heard: Heard, _deps: Any) -> Speaker | None:
        if not heard.text or not heard.roster:
            return None

        words = heard.text.split()
        window = [tidy(word) for word in words[: self.cfg.name_window_words]]
        if not window:
            return None

        cued = self._cue_in(" ".join(window))
        best = self._best_match(window, heard.roster)
        if best is None:
            return None

        score, at, size, row = best
        needed = self.cfg.name_match if cued else self.cfg.name_match + self.cfg.no_cue_extra
        if score < needed or (not cued and at > 0):
            # Without a lead-in, only a name right at the front counts. A
            # lesson about Meera's garden is not Meera speaking.
            return None

        text = self._without_name(words, at, size) if self.cfg.strip_name else heard.text
        return Speaker(student_id=row["id"], name=row["name"], how=self.name, text=text)

    def _cue_in(self, start: str) -> bool:
        return any(cue.lower() in start for cue in self.cfg.name_cues)

    def _best_match(self, window: list[str], roster: list[dict]):
        best = None
        for size in range(1, WORDS + 1):
            for at in range(len(window) - size + 1):
                said = " ".join(window[at : at + size])
                if not said:
                    continue
                for row in roster:
                    full = row["name"].lower()
                    for target in {full, full.split()[0]}:
                        score = looks_like(said, target)
                        if best is None or score > best[0]:
                            best = (score, at, size, row)
        return best

    def _without_name(self, words: list[str], at: int, size: int) -> str:
        rest = " ".join(words[at + size :]).lstrip(PUNCTUATION + " ")
        for lead in sorted(self.cfg.question_lead_ins, key=len, reverse=True):
            pattern = re.compile(rf"^{re.escape(lead)}\b[\s,]*", re.IGNORECASE)
            if pattern.match(rest):
                rest = pattern.sub("", rest, count=1)
                break
        return rest.strip() or " ".join(words).strip()
