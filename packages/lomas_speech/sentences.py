from __future__ import annotations

import re

# Splitting a paragraph into sentences, so the robot can stop at the end of
# one. This is not linguistics: it is the difference between a teacher
# pausing and a machine being switched off mid-word.

# Devanagari's danda counts, and so do the three obvious ones.
ENDINGS = re.compile(r"(?<=[.!?।])\s+")
# Not an ending: an initial, a title, or a number with a decimal point in
# it. Splitting at one of these makes the robot pause in the middle of
# somebody's name.
INITIAL = re.compile(r"\b[A-Z]\.$")
DECIMAL = re.compile(r"\d\.$")
TITLE = re.compile(r"\b(?:Dr|Mr|Mrs|Ms|Prof|Sri|Smt|St|Jr|Sr|vs|etc|No|Fig)\.$", re.IGNORECASE)
ONE = 1


def split(text: str) -> list[str]:
    """Whole sentences, in order, with the punctuation kept.

    A text with no full stop in it comes back as one sentence, which is
    correct: there is nowhere to stop politely.
    """
    body = " ".join(text.split())
    if not body:
        return []

    sentences: list[str] = []
    for piece in ENDINGS.split(body):
        if sentences and _joins_on(sentences[-ONE]):
            sentences[-ONE] = f"{sentences[-ONE]} {piece}"
            continue
        sentences.append(piece)
    return [sentence for sentence in sentences if sentence.strip()]


def _joins_on(previous: str) -> bool:
    """Whether the last piece did not really end. "Dr." and "3." are not the
    end of a sentence, and splitting there makes the robot pause mid-name."""
    return bool(INITIAL.search(previous) or DECIMAL.search(previous) or TITLE.search(previous))
