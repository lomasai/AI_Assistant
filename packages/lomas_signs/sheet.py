from __future__ import annotations

import numpy as np

from lomas_core.errors import LomasError

# A class's cards, on paper. Nothing is bought and nothing is downloaded:
# the markers are generated here and printed on whatever printer the school
# has. Matte paper, not glossy - gloss throws a highlight into the camera
# and the detector loses the square.

WHITE = 255
BLACK = 0
CHANNELS = 3
FONT_SCALE_DIVISOR = 420.0
NAME_SCALE_DIVISOR = 900.0
NAME_WORDS = 2
LINE = 2
HALF = 2


def capacity(dictionary: str) -> int:
    """How many different cards this dictionary can make.

    DICT_4X4_50 is fifty, which is a class. A school handing cards to four
    hundred children wants DICT_4X4_250 or a 5x5 family, and a bigger
    printed card to go with it.
    """
    import cv2

    family = getattr(cv2.aruco, dictionary, None)
    if family is None:
        raise LomasError(f"no marker dictionary '{dictionary}'")
    return int(cv2.aruco.getPredefinedDictionary(family).bytesList.shape[0])


def marker(dictionary: str, marker_id: int, pixels: int) -> np.ndarray:
    """One marker, as a square greyscale image."""
    import cv2

    family = getattr(cv2.aruco, dictionary, None)
    if family is None:
        raise LomasError(f"no marker dictionary '{dictionary}'")
    return cv2.aruco.generateImageMarker(
        cv2.aruco.getPredefinedDictionary(family), marker_id, pixels
    )


def card(dictionary: str, marker_id: int, name: str, pixels: int, answers: list[str],
         quiet_px: int = 0) -> np.ndarray:
    """One child's card: the marker, a quiet border, the name, and a letter
    along each edge so a child can see which way up they are holding it.

    The quiet border is not decoration. A marker printed to the edge of the
    paper is a marker the detector will not find.
    """
    import cv2

    # Wide on purpose: the letters live in it, and a marker printed to the
    # edge of the paper is a marker the detector will not find.
    border = quiet_px or max(1, pixels // 4)
    side = pixels + border * HALF
    face = np.full((side, side, CHANNELS), WHITE, np.uint8)
    square = cv2.cvtColor(marker(dictionary, marker_id, pixels), cv2.COLOR_GRAY2BGR)
    face[border:border + pixels, border:border + pixels] = square

    # Small, and in the bottom-left corner: the middle of that edge belongs
    # to an answer letter, and a full name written across it covered the
    # letter on the first sheet that came off the printer.
    _write(face, _short(name) or f"#{marker_id}", (border // HALF, side - border // 5),
           side / NAME_SCALE_DIVISOR)
    scale = side / FONT_SCALE_DIVISOR
    for turn, letter in enumerate(answers[:4]):
        _edge_letter(face, letter, turn, side, border, scale)
    return face


def _edge_letter(face: np.ndarray, letter: str, turn: int, side: int, border: int,
                 scale: float) -> None:
    """The letter that is at the top when the card is held that way up.

    Rotating the picture rather than placing four differently-oriented bits
    of text: the letter has to read the right way up to the child holding
    it, whichever edge they have chosen.
    """
    import cv2

    for _ in range(turn):
        face[:] = cv2.rotate(face, cv2.ROTATE_90_COUNTERCLOCKWISE)
    _write(face, letter, (side // HALF, int(border * 0.7)), scale * HALF)
    for _ in range(turn):
        face[:] = cv2.rotate(face, cv2.ROTATE_90_CLOCKWISE)


def _write(face: np.ndarray, text: str, at: tuple[int, int], scale: float) -> None:
    import cv2

    cv2.putText(face, text, at, cv2.FONT_HERSHEY_SIMPLEX, scale, (BLACK, BLACK, BLACK), LINE,
                cv2.LINE_AA)


def _short(name: str) -> str:
    """Enough for a teacher to hand the right card to the right child, and
    short enough not to run into the answer letter below the marker."""
    words = name.split()
    if len(words) <= NAME_WORDS:
        return name
    return " ".join(words[:NAME_WORDS]) + f" {words[NAME_WORDS][0]}."


def sheet(cards: list[np.ndarray], across: int, gap_px: int) -> np.ndarray:
    """A page of cards, laid out to be cut up."""
    if not cards:
        raise LomasError("no cards to print")

    side = max(card.shape[0] for card in cards)
    down = (len(cards) + across - 1) // across
    page = np.full((down * (side + gap_px) + gap_px,
                    across * (side + gap_px) + gap_px, CHANNELS), WHITE, np.uint8)
    for index, one in enumerate(cards):
        row, column = divmod(index, across)
        top = gap_px + row * (side + gap_px)
        left = gap_px + column * (side + gap_px)
        page[top:top + one.shape[0], left:left + one.shape[1]] = one
    return page
