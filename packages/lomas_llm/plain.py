from __future__ import annotations

import re

# Models write typography. A voice reads it badly and a Pi console without a
# UTF-8 locale prints it as "?": "Don?t", "CO?", "carbon?dioxide" were all in
# one class on the robot. None of these carry meaning plain ASCII does not.
TYPOGRAPHY = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "′": "'",
    "“": '"', "”": '"', "„": '"', "″": '"',
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": " - ",
    "…": "...", " ": " ", " ": " ", " ": " ",
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
    "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
    "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4",
    "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9",
})

# Markdown a board cannot render and a voice would read out as symbols.
EMPHASIS = re.compile(r"(\*\*|__|\*|`)")
SPACES = re.compile(r"[ \t]+")
BREAKS = re.compile(r"\s*\n\s*")


def plain_text(text: str) -> str:
    """What a model said, as something to speak and show.

    Line breaks become spaces: the model breaks lines for a chat window, and
    a robot reading a paragraph has no use for them. Other scripts - Hindi,
    Marathi - pass through untouched; only typography is replaced.
    """
    text = EMPHASIS.sub("", text.translate(TYPOGRAPHY))
    return SPACES.sub(" ", BREAKS.sub(" ", text)).strip()
