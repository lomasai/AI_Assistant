from __future__ import annotations

import threading

from lomas_core import logging as log
from lomas_core.errors import LomasError
from lomas_core.schema import Config

from app.face.state import (
    ASKING,
    AWAY,
    LISTENING,
    SLEEPING,
    SPEAKING,
    THINKING,
    FaceState,
)
from app.face.surface import FACE_SURFACES

# The face, as numbers. A 7-inch panel seen from a desk two metres away, so
# everything is large and there is very little of it.
BACKGROUND = (14, 18, 28)
EYE = (232, 238, 250)
PUPIL = (24, 30, 44)
MOUTH = (232, 238, 250)
TEXT = (226, 232, 244)
DIM = (120, 132, 156)
MOODS = {"engaged": (86, 196, 134), "drifting": (232, 184, 84),
         "speaking": (110, 168, 254), AWAY: (92, 100, 120)}

BLINK_EVERY = 4.2
BLINK_FOR = 0.12
TALK_HZ = 6.0
IDLE_FPS = 10
LINE_SHARE = 0.86  # of the screen width, before a line wraps
DEFAULT_DISPLAY = ":0"  # the Pi's own desktop, when nothing says otherwise


@FACE_SURFACES.register("pygame")
class PygameFace:
    """The robot's face, drawn directly instead of through a browser.

    Chromium showing one face costs a Pi several hundred megabytes and a
    steady share of a core it would rather spend on the detector. This is the
    same face in about fifty megabytes: two eyes, a mouth that moves while
    the robot speaks, one line of text and a row of names.

    It runs on its own thread and reads a snapshot, so a slow frame cannot
    hold up a lesson, and a missing pygame is a robot with no face rather
    than a robot that will not start.
    """

    name = "pygame"

    def __init__(self, cfg: Config, state: FaceState) -> None:
        self.cfg = cfg
        self.screen = cfg.display.face_screen
        self.state = state
        self.log = log.get("face")
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        try:
            import pygame  # noqa: F401
        except ImportError as exc:
            raise LomasError(
                "pygame is not installed, so there is no face to show. "
                "pip install pygame, or set display.face_screen.surface: browser."
            ) from exc

        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="face", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.cfg.web.shutdown_seconds)
            self._thread = None

    # --- the window -------------------------------------------------------

    def _run(self) -> None:
        import pygame

        try:
            opened = open_display(pygame, self.screen)
        except LomasError as exc:
            # The one that cost an evening: pygame.init() reports nothing when
            # the video system fails, and the first call that needs a screen
            # dies instead, four frames from the cause.
            self.log.error("no face: %s", exc)
            return

        pygame.font.init()
        flags = pygame.FULLSCREEN if self.screen.fullscreen else 0
        surface = pygame.display.set_mode((self.screen.width, self.screen.height), flags)
        pygame.display.set_caption("LomasAI")
        pygame.mouse.set_visible(False)
        self.log.info("face on %sx%s through %s", self.screen.width, self.screen.height, opened)

        big = pygame.font.Font(None, int(self.cfg.display.base_font_px * self.screen.scale))
        small = pygame.font.Font(None, int(self.cfg.display.base_font_px * 0.6 * self.screen.scale))
        clock = pygame.time.Clock()

        try:
            while not self._stop.is_set():
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self._stop.set()
                self._draw(pygame, surface, big, small)
                pygame.display.flip()
                clock.tick(IDLE_FPS)
        except Exception as exc:  # a display that goes away is not a lesson ending
            self.log.error("the face stopped: %s", exc)
        finally:
            pygame.quit()

    def _draw(self, pygame, surface, big, small) -> None:
        import time

        look = self.state.snapshot()
        width, height = surface.get_size()
        surface.fill(BACKGROUND)

        now = time.monotonic()
        self._eyes(pygame, surface, look, now, width, height)
        self._mouth(pygame, surface, look, now, width, height)
        self._words(surface, big, small, look, width, height)
        self._ribbon(pygame, surface, small, look, width, height)

    def _eyes(self, pygame, surface, look, now, width, height) -> None:
        radius = int(height * 0.11)
        top = int(height * 0.3)
        blinking = (now % BLINK_EVERY) < BLINK_FOR or look.state == SLEEPING

        for side in (-1, 1):
            middle = (width // 2 + side * int(width * 0.16), top)
            if blinking:
                pygame.draw.line(surface, EYE, (middle[0] - radius, top),
                                 (middle[0] + radius, top), max(3, radius // 5))
                continue
            pygame.draw.circle(surface, EYE, middle, radius)
            # The pupil drifts up when the robot is thinking, which reads as
            # thinking to a child and costs one line.
            lift = int(radius * 0.3) if look.state == THINKING else 0
            pygame.draw.circle(surface, PUPIL, (middle[0], top - lift), int(radius * 0.45))

    def _mouth(self, pygame, surface, look, now, width, height) -> None:
        import math

        middle_x, middle_y = width // 2, int(height * 0.5)
        span = int(width * 0.18)

        if look.state == SPEAKING:
            open_by = abs(math.sin(now * TALK_HZ)) * height * 0.06 + height * 0.01
            pygame.draw.ellipse(
                surface, MOUTH,
                (middle_x - span // 2, middle_y - open_by / 2, span, open_by))
            return

        if look.state in (LISTENING, ASKING):
            pygame.draw.arc(surface, MOUTH,
                            (middle_x - span // 2, middle_y - span // 4, span, span // 2),
                            3.34, 6.08, max(3, height // 120))
            return

        pygame.draw.line(surface, MOUTH, (middle_x - span // 3, middle_y),
                         (middle_x + span // 3, middle_y), max(3, height // 140))

    def _words(self, surface, big, small, look, width, height) -> None:
        if not look.line:
            return

        top = int(height * 0.62)
        for line in wrap(look.line, big, int(width * LINE_SHARE)):
            drawn = big.render(line, True, TEXT)
            surface.blit(drawn, drawn.get_rect(center=(width // 2, top)))
            top += drawn.get_height() + 4

        for number, option in enumerate(look.options, start=1):
            drawn = small.render(f"{number}.  {option}", True, DIM)
            surface.blit(drawn, (int(width * 0.1), top))
            top += drawn.get_height() + 2

    def _ribbon(self, pygame, surface, small, look, width, height) -> None:
        if not look.children:
            return

        left = int(width * 0.04)
        bottom = height - int(height * 0.07)
        for child in look.children.values():
            colour = MOODS.get(child.mood, MOODS[AWAY])
            pygame.draw.circle(surface, colour, (left + 6, bottom), 6)
            drawn = small.render(child.name, True, DIM if child.mood == AWAY else TEXT)
            surface.blit(drawn, (left + 18, bottom - drawn.get_height() // 2))
            left += drawn.get_width() + 44


def open_display(pygame, screen) -> str:
    """Get a screen, and say which one, or say why there is none.

    A Pi shows this face three ways: through the desktop over X11, straight
    at the panel with no desktop at all, and through a VNC session that has
    its own display. Rather than ask which, each is tried in turn - the whole
    point of the face is that the robot boots and is there.
    """
    import os

    wanted = [screen.driver] if screen.driver else ["", "x11", "kmsdrm", "fbcon"]
    tried = []
    for driver in wanted:
        if driver:
            os.environ["SDL_VIDEODRIVER"] = driver
        elif not os.environ.get("DISPLAY"):
            # A terminal over ssh has no DISPLAY; the Pi's own desktop is :0
            # and is what somebody sitting in front of the robot is looking at.
            os.environ["DISPLAY"] = DEFAULT_DISPLAY

        try:
            pygame.display.init()
            return driver or pygame.display.get_driver()
        except pygame.error as exc:
            tried.append(f"{driver or 'default'} ({exc})")
            pygame.display.quit()
            os.environ.pop("SDL_VIDEODRIVER", None)

    raise LomasError(
        "no screen to draw a face on: " + "; ".join(tried) + ". On a Pi with a "
        "desktop, run the robot from that desktop; without one, set "
        "display.face_screen.driver: kmsdrm. Or browser, for a tab instead."
    )


def wrap(text: str, font, room: int) -> list[str]:
    """One line of speech, broken to fit. A face has no scrollbar."""
    lines: list[str] = []
    line = ""
    for word in text.split():
        candidate = f"{line} {word}".strip()
        if font.size(candidate)[0] <= room or not line:
            line = candidate
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines
