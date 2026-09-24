from __future__ import annotations

import math
import threading
from pathlib import Path

from lomas_core import logging as log
from lomas_core.errors import LomasError
from lomas_core.schema import Config

from lomas_core.contracts import VOLUME_CHANGED, VolumeChanged

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

FACE_BUTTONS = "the robot's own screen"

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

# Where the parts of the face sit, as shares of the screen, so a 7-inch
# panel and a 24-inch monitor show the same face rather than the same pixels.
EYES_AT = 0.32
EYE_SHARE = 0.12
EYE_APART = 0.17
PUPIL_SHARE = 0.45
GLINT_SHARE = 0.3
MOUTH_AT = 0.52
MOUTH_SPAN = 0.2
TEXT_AT = 0.7      # below the mouth, not across it
OPTIONS_GAP = 6

BLINK_EVERY = 4.2
BLINK_FOR = 0.12
BLINK_CURVE = 0.18   # how much a blinking eye curves
SLEEP_CURVE = 0.42   # ...and a sleeping one, which is the difference between
LID_THICKNESS = 7    #    resting and switched off
LOOK_AROUND_HZ = 0.9
LOOK_BY = 0.25
LOOK_UP = 0.3
SMILE_FROM = 0.5
MOUTH_THICK = 60
MOUTH_THIN = 160
TALK_OPEN = 0.06
TALK_REST = 0.012
TALK_HZ = 6.0
IDLE_FPS = 10
LINE_SHARE = 0.86  # of the screen width, before a line wraps
WAITING = "waiting for a class"

# The volume control, in the corner where a hand rests rather than across
# the face. Quieter, louder, mute.
KNOB = (36, 44, 62)
KNOB_LIT = (110, 168, 254)
KNOB_MUTED = (180, 68, 52)
KNOB_TEXT = (226, 232, 244)
KNOB_RADIUS = 14
BAR_HEIGHT = 10
BAR_WIDTH_SHARE = 0.34
QUIETER, LOUDER, MUTE = "-", "+", "M"
ONE_STEP = 1.0
FULL = 100
RUNTIME_DIR = "XDG_RUNTIME_DIR"  # where Wayland leaves its socket
WAYLAND_SOCKETS = "wayland-*"
LOCK = ".lock"  # beside each socket, and not itself a screen
X_SOCKETS = "/tmp/.X11-unix"
X_DISPLAYS = "X*"
NOT_AVAILABLE = "not available"  # SDL's words for a backend it does not have


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

    def __init__(self, cfg: Config, state: FaceState, volume=None) -> None:
        self.cfg = cfg
        self.screen = cfg.display.face_screen
        self.state = state
        # The robot's own volume knob. None is a face with no controls on
        # it, which is what a browser surface and a test both want.
        self.volume = volume
        self.log = log.get("face")
        self._stop = threading.Event()
        self._full = cfg.display.face_screen.fullscreen
        self._touched = 0.0
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
        self._full = self.screen.fullscreen
        flags = pygame.FULLSCREEN if self._full else 0
        surface = pygame.display.set_mode((self.screen.width, self.screen.height), flags)
        pygame.display.set_caption("LomasAI")
        # Visible where there is a volume control to point at: an invisible
        # cursor over a button nobody can see is two problems.
        pygame.mouse.set_visible(bool(self._knob() and self.screen.show_volume))
        self.log.info("face on %sx%s through %s - Esc leaves fullscreen, Q closes it",
                      self.screen.width, self.screen.height, opened)

        big = pygame.font.Font(None, int(self.cfg.display.base_font_px * self.screen.scale))
        small = pygame.font.Font(None, int(self.cfg.display.base_font_px * 0.6 * self.screen.scale))
        clock = pygame.time.Clock()

        try:
            while not self._stop.is_set():
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self._stop.set()
                    elif event.type == pygame.KEYDOWN:
                        surface = self._keyed(pygame, event.key, surface)
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        # A tap on the panel, or a click over VNC. Same
                        # three targets either way.
                        self._pressed(event.pos, surface.get_size())
                self._draw(pygame, surface, big, small)
                pygame.display.flip()
                clock.tick(IDLE_FPS)
        except pygame.error as exc:
            # The window closing while a frame is being drawn is the robot
            # shutting down, not a fault worth a line at a teacher.
            self.log.debug("the face closed: %s", exc)
        except Exception as exc:  # a display that goes away is not a lesson ending
            self.log.error("the face stopped: %s", exc)
        finally:
            pygame.quit()

    def _keyed(self, pygame, key, surface):
        """Escape leaves fullscreen; Q closes the face; up, down and M are
        the volume.

        A face that covers the whole screen with no way back is a robot you
        have to kill from another terminal, which is exactly the wrong thing
        to be doing in front of a class.
        """
        if key == pygame.K_q:
            self._stop.set()
            return surface
        if key in (pygame.K_UP, pygame.K_DOWN):
            self._turn(ONE_STEP if key == pygame.K_UP else -ONE_STEP)
            return surface
        if key == pygame.K_m:
            self._mute()
            return surface
        if key not in (pygame.K_ESCAPE, pygame.K_f):
            return surface

        self._full = not self._full
        return pygame.display.set_mode((self.screen.width, self.screen.height),
                                       pygame.FULLSCREEN if self._full else 0)

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
        self._volume(pygame, surface, small, now, width, height)

    def _eyes(self, pygame, surface, look, now, width, height) -> None:
        """Two eyes that blink, and close rather than vanish.

        Drawn as rounded shapes: a straight bar across a dark screen reads as
        a broken screen, and a child looking at a robot should see a face
        before they see anything else.
        """
        import math

        radius = int(height * EYE_SHARE)
        top = int(height * EYES_AT)
        asleep = look.state == SLEEPING
        blinking = asleep or (now % BLINK_EVERY) < BLINK_FOR

        for side in (-1, 1):
            middle_x = width // 2 + side * int(width * EYE_APART)
            if blinking:
                self._closed_eye(pygame, surface, middle_x, top, radius, asleep)
                continue

            pygame.draw.circle(surface, EYE, (middle_x, top), radius)
            # The pupil looks about while thinking and settles while speaking,
            # which reads as thinking and speaking without a word of text.
            drift = math.sin(now * LOOK_AROUND_HZ) if look.state == THINKING else 0.0
            across = int(drift * radius * LOOK_BY)
            lift = int(radius * LOOK_UP) if look.state == THINKING else 0
            pygame.draw.circle(surface, PUPIL, (middle_x + across, top - lift),
                               int(radius * PUPIL_SHARE))
            # One highlight. It is the difference between an eye and a dot.
            glint = int(radius * GLINT_SHARE)
            pygame.draw.circle(surface, EYE, (middle_x + across + glint, top - lift - glint),
                               max(2, glint // 2))

    def _closed_eye(self, pygame, surface, middle_x, top, radius, asleep) -> None:
        """A closed eye curves. Asleep it curves more, and that is the whole
        difference between a robot resting and a robot switched off."""
        depth = radius * (SLEEP_CURVE if asleep else BLINK_CURVE)
        thickness = max(3, radius // LID_THICKNESS)
        for step in range(thickness):
            box = (middle_x - radius, int(top - depth) + step, radius * 2, int(depth * 2))
            pygame.draw.arc(surface, EYE, box, math.pi, math.tau, 2)

    def _mouth(self, pygame, surface, look, now, width, height) -> None:
        import math

        middle_x, middle_y = width // 2, int(height * MOUTH_AT)
        span = int(width * MOUTH_SPAN)

        if look.state == SPEAKING:
            # Open and shut, rounded, so it reads as talking rather than as a
            # bar changing height.
            open_by = abs(math.sin(now * TALK_HZ)) * height * TALK_OPEN + height * TALK_REST
            pygame.draw.ellipse(
                surface, MOUTH,
                (middle_x - span // 2, int(middle_y - open_by / 2), span, int(open_by)))
            return

        if look.state == SLEEPING:
            # A small, level line: resting, not unhappy.
            pygame.draw.line(surface, DIM, (middle_x - span // 4, middle_y),
                             (middle_x + span // 4, middle_y), max(3, height // MOUTH_THIN))
            return

        # Listening, thinking, asking: a slight smile, drawn as a few arcs
        # stacked. One arc at any thickness comes out as a hairline with
        # gaps in it, which on a face reads as a fault.
        thickness = max(3, height // MOUTH_THICK)
        for step in range(thickness):
            box = (middle_x - span // 2, middle_y - span // 4 + step, span, span // 2)
            pygame.draw.arc(surface, MOUTH, box, math.pi + SMILE_FROM, math.tau - SMILE_FROM, 2)

    def _words(self, surface, big, small, look, width, height) -> None:
        """What is being said, under the face rather than through it."""
        top = int(height * TEXT_AT)
        if not look.line:
            # A dark screen with a sleeping face on it should still say what
            # it is waiting for.
            if look.state == SLEEPING:
                drawn = small.render(WAITING, True, DIM)
                surface.blit(drawn, drawn.get_rect(center=(width // 2, top)))
            return

        for line in wrap(look.line, big, int(width * LINE_SHARE)):
            drawn = big.render(line, True, TEXT)
            surface.blit(drawn, drawn.get_rect(center=(width // 2, top)))
            top += drawn.get_height() + OPTIONS_GAP

        for number, option in enumerate(look.options, start=1):
            drawn = small.render(f"{number}.  {option}", True, DIM)
            surface.blit(drawn, drawn.get_rect(center=(width // 2, top)))
            top += drawn.get_height() + OPTIONS_GAP

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


    # --- the volume, on the robot's own screen ----------------------------

    def _knob(self):
        """The control, if this robot has one. A face is a face without it."""
        return self.volume

    def _buttons(self, width: int, height: int) -> list[tuple[str, tuple[int, int]]]:
        """Three targets along the bottom right, sized for a finger.

        Bottom right because that is where a hand rests on a panel on a desk,
        and away from the names along the bottom left.
        """
        size = self.screen.volume_button_px
        margin = self.screen.volume_margin_px
        y = height - margin - size // 2
        return [
            (MUTE, (width - margin - size // 2 - (size + margin) * 2, y)),
            (QUIETER, (width - margin - size // 2 - (size + margin), y)),
            (LOUDER, (width - margin - size // 2, y)),
        ]

    def _pressed(self, where, size) -> None:
        knob = self._knob()
        if knob is None or not self.screen.show_volume:
            return

        reach = self.screen.volume_button_px // 2
        for label, (x, y) in self._buttons(*size):
            if abs(where[0] - x) <= reach and abs(where[1] - y) <= reach:
                if label == MUTE:
                    self._mute()
                else:
                    self._turn(ONE_STEP if label == LOUDER else -ONE_STEP)
                return

    def _turn(self, steps: float) -> None:
        knob = self._knob()
        if knob is None:
            return
        knob.nudge(steps)
        self._moved(knob)

    def _mute(self) -> None:
        knob = self._knob()
        if knob is None:
            return
        knob.mute(not knob.muted)
        self._moved(knob)

    def _moved(self, knob) -> None:
        """Said out loud and put on the bus.

        "The buttons do nothing" and "the buttons work and the robot is
        inaudible anyway" look identical from across a room, and only one of
        them is a bug in this program. This is how a trace tells them apart.
        """
        import time

        self._touched = time.monotonic()
        self.log.info("volume %s", knob.describe())
        self.state.bus.publish(VOLUME_CHANGED, VolumeChanged(
            level=knob.level, muted=knob.muted, by=FACE_BUTTONS,
            describe=knob.describe(), at=time.time()))

    def _volume(self, pygame, surface, small, now, width, height) -> None:
        knob = self._knob()
        if knob is None or not self.screen.show_volume:
            return

        for label, (x, y) in self._buttons(width, height):
            lit = KNOB_MUTED if (label == MUTE and knob.muted) else KNOB
            pygame.draw.circle(surface, lit, (x, y), self.screen.volume_button_px // 2)
            glyph = small.render(label, True, KNOB_TEXT)
            surface.blit(glyph, glyph.get_rect(center=(x, y)))

        # The bar only while somebody is changing it. A number on a face all
        # day makes the robot a control panel with eyes.
        if now - self._touched > self.screen.volume_shown_seconds:
            return

        bar = int(width * BAR_WIDTH_SHARE)
        left = (width - bar) // 2
        top = height - self.screen.volume_margin_px * 2 - BAR_HEIGHT
        pygame.draw.rect(surface, KNOB, (left, top, bar, BAR_HEIGHT),
                         border_radius=BAR_HEIGHT // 2)
        filled = 0 if knob.muted else int(bar * knob.level)
        if filled:
            pygame.draw.rect(surface, KNOB_LIT, (left, top, filled, BAR_HEIGHT),
                             border_radius=BAR_HEIGHT // 2)
        said = "muted" if knob.muted else f"{round(knob.level * FULL)}%"
        label = small.render(said, True, DIM)
        surface.blit(label, label.get_rect(center=(width // 2, top - KNOB_RADIUS)))


def running_sessions() -> list[tuple[str, dict]]:
    """The desktops this Pi is actually running, found rather than guessed.

    A terminal over ssh has no screen of its own, but the screen somebody is
    looking at is right there and belongs to the same user. Wayland leaves a
    socket in the runtime directory and X leaves one in /tmp/.X11-unix, so
    both can be named exactly instead of being tried by folklore.
    """
    import os

    found = []
    runtime = os.environ.get(RUNTIME_DIR, "")
    if not runtime and hasattr(os, "getuid"):
        runtime = f"/run/user/{os.getuid()}"
    if runtime and Path(runtime).is_dir():
        for socket in sorted(Path(runtime).glob(WAYLAND_SOCKETS)):
            if socket.suffix != LOCK:
                found.append(("wayland", {RUNTIME_DIR: runtime, "WAYLAND_DISPLAY": socket.name}))

    sockets = Path(X_SOCKETS)
    if sockets.is_dir():
        for socket in sorted(sockets.glob(X_DISPLAYS)):
            display = ":" + socket.name[1:]
            found += [("x11", {"DISPLAY": display, "XAUTHORITY": cookie})
                      for cookie in cookies()]
    return found


def cookies() -> list[str]:
    """Every X cookie this machine might be using, and then none at all.

    X refuses a client that cannot prove which session it belongs to, and a
    robot started over ssh has to borrow the desktop's proof. Where that is
    kept depends on what started the desktop, so the ones that exist are all
    tried rather than argued about.
    """
    import os

    where = [
        os.environ.get("XAUTHORITY", ""),
        os.path.expanduser("~/.Xauthority"),
        f"/run/user/{os.getuid()}/gdm/Xauthority" if hasattr(os, "getuid") else "",
        f"/run/user/{os.getuid()}/xauth_for_lightdm" if hasattr(os, "getuid") else "",
    ]
    found = [path for path in where if path and Path(path).exists()]
    # The empty one last: some servers let a local user straight in, and a
    # wrong cookie is refused where no cookie would have been let through.
    return [*dict.fromkeys(found), ""]


def attempts(screen) -> list[tuple[str, dict]]:
    """Every way this Pi has of putting something on a screen, in order.

    The terminal's own display first - that is a robot started from the
    desktop, which is the normal case - then the sessions actually running
    on the machine, then the panel itself with no desktop at all.
    """
    if screen.driver:
        return [(screen.driver, {})]
    return [("", {}), *running_sessions(), ("kmsdrm", {}), ("fbcon", {})]


def open_display(pygame, screen) -> str:
    """Get a screen, and say which one, or say why there is none.

    Rather than ask which kind of Pi this is, each is tried in turn: the
    whole point of a face the robot draws itself is that it boots and is
    there.
    """
    import os

    tried = []
    for driver, environment in attempts(screen):
        before = {name: os.environ.get(name) for name in ("SDL_VIDEODRIVER", *environment)}
        if driver:
            os.environ["SDL_VIDEODRIVER"] = driver
        os.environ.update(environment)

        try:
            pygame.display.init()
            return driver or pygame.display.get_driver()
        except pygame.error as exc:
            shown = " ".join(f"{k}={v}" for k, v in environment.items())
            tried.append(f"{driver or 'this terminal'}{' ' + shown if shown else ''}: {exc}")
            pygame.display.quit()
            for name, value in before.items():
                os.environ.pop(name, None) if value is None else os.environ.update({name: value})

    # Every driver refusing by name is a different fault from every display
    # refusing: the first is an SDL with nothing to draw through, which the
    # Pi's packaged pygame turned out to be while X itself was answering
    # xdpyinfo perfectly well.
    if all(NOT_AVAILABLE in line for line in tried):
        raise LomasError(
            "this pygame has no way to draw at all (" + "; ".join(tried) + "). The one "
            "from apt can be built without any video backend: pip install --upgrade "
            "pygame inside the virtualenv, which brings its own. Or surface: browser."
        )

    raise LomasError(
        "no screen to draw a face on. Tried " + "; ".join(tried) + ". Run the robot "
        "from a terminal on the Pi's own screen (or in the VNC session) so it can "
        "see that display; on a Pi with no desktop at all set "
        "display.face_screen.driver: kmsdrm. Or surface: browser, for a tab."
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
