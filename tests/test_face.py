"""The robot's own face, drawn without a browser.

Chromium showing one face costs a Pi several hundred megabytes and a share
of a core the detector wants. What the face should look like is decided from
events and tested here without a screen; the window that draws it is checked
against a real pygame surface, off-screen.
"""
from __future__ import annotations

import os

import pytest

from lomas_core.clock import FakeClock
from lomas_core.config import load
from lomas_core.contracts import (
    ATTENDANCE_MARKED,
    LESSON_SEGMENT,
    QUESTION_ASKED,
    QUIZ_POSED,
    ROBOT_SAY,
    ROBOT_SPOKE,
    ROBOT_STATE,
    SESSION_CLOSED,
    SESSION_OPENED,
    AttendanceMarked,
    LessonSegment,
    QuestionAsked,
    QuizPosed,
    SessionClosed,
    SessionOpened,
    StudentIdentified,
    Utterance,
)
from lomas_core.contracts import STUDENT_IDENTIFIED

from app import container
from app.face import FACE_SURFACES, FaceState

HEADLESS = [
    "storage.backend=memory",
    "vision.pipeline.enabled=false",
    "hardware.enabled=false",
    "speech.tts.engine=null",
    "speech.stt.engine=keyboard",
    "speech.wake.engine=keyboard",
    "llm.provider=offline",
]


@pytest.fixture
def bus():
    """The bus on its own. The face subscribes and nothing else, which is
    exactly what makes it testable without a system around it."""
    return container.event_bus(load("config", "debug", HEADLESS, use_env=False))


@pytest.fixture
def face(bus):
    return FaceState(bus)


def opened(bus) -> None:
    bus.publish(SESSION_OPENED, SessionOpened(
        session_id="s1", org_id="o", school_id="s", class_id="c",
        language="en", topic="photosynthesis", started_at=0.0))


# --- what the face is doing -----------------------------------------------


def test_a_robot_with_no_class_is_asleep(face) -> None:
    assert face.snapshot().state == "sleeping"


def test_a_class_wakes_it_up(bus, face) -> None:
    opened(bus)
    assert face.snapshot().state == "listening"


def test_speaking_shows_the_words_and_moves_the_mouth(bus, face) -> None:
    opened(bus)
    bus.publish(ROBOT_SAY, Utterance(text="A leaf needs three things.", language="en"))

    look = face.snapshot()
    assert look.state == "speaking"
    assert look.line == "A leaf needs three things."


def test_a_question_makes_it_think(bus, face) -> None:
    opened(bus)
    bus.publish(QUESTION_ASKED, QuestionAsked(session_id="s1", text="why are leaves green?"))

    assert face.snapshot().state == "thinking"


def test_a_quiz_question_shows_its_options(bus, face) -> None:
    opened(bus)
    bus.publish(QUIZ_POSED, QuizPosed(session_id="s1", question_id="q1",
                                      text="Which gas does a leaf let out?",
                                      options=("Carbon dioxide", "Oxygen")))

    look = face.snapshot()
    assert look.state == "asking"
    assert look.options == ("Carbon dioxide", "Oxygen")


def test_the_board_line_is_preferred_to_the_spoken_one(bus, face) -> None:
    """A segment written for a screen reads differently from one read out."""
    opened(bus)
    bus.publish(LESSON_SEGMENT, LessonSegment(
        session_id="s1", lesson_id="photosynthesis", segment_id="s1", index=0, total=6,
        say="Every animal has to find food, and a plant never goes anywhere.",
        display="Plants make their own food"))

    assert face.snapshot().line == "Plants make their own food"


def test_it_settles_back_after_speaking(bus, face) -> None:
    opened(bus)
    bus.publish(ROBOT_SAY, Utterance(text="hello", language="en"))
    bus.publish(ROBOT_SPOKE, Utterance(text="hello", language="en"))

    assert face.snapshot().state == "listening"


def test_the_class_goes_home(bus, face) -> None:
    opened(bus)
    bus.publish(ATTENDANCE_MARKED, AttendanceMarked(
        session_id="s1", student_id="a", name="Ananya Sharma", source="roster"))
    bus.publish(SESSION_CLOSED, SessionClosed(session_id="s1", ended_at=1.0, reason="closed"))

    look = face.snapshot()
    assert look.state == "sleeping"
    assert look.children == {}


# --- the row of names -----------------------------------------------------


def test_a_child_appears_by_first_name_only(bus, face) -> None:
    opened(bus)
    bus.publish(ATTENDANCE_MARKED, AttendanceMarked(
        session_id="s1", student_id="a", name="Ananya Sharma", source="roster"))

    assert [child.name for child in face.snapshot().children.values()] == ["Ananya"]


def test_being_seen_changes_the_dot(bus, face) -> None:
    opened(bus)
    bus.publish(ATTENDANCE_MARKED, AttendanceMarked(
        session_id="s1", student_id="a", name="Ananya Sharma", source="roster"))
    assert face.snapshot().children["a"].mood == "away"

    bus.publish(STUDENT_IDENTIFIED, StudentIdentified(
        student_id="a", track_id=1, source_id="head", zone="front", at=1.0))
    assert face.snapshot().children["a"].mood == "engaged"


def test_the_child_being_spoken_to_is_marked(bus, face) -> None:
    opened(bus)
    for who, name in (("a", "Ananya Sharma"), ("b", "Rahul Deshmukh")):
        bus.publish(ATTENDANCE_MARKED, AttendanceMarked(
            session_id="s1", student_id=who, name=name, source="roster"))

    bus.publish(ROBOT_SAY, Utterance(text="what do you think?", language="en",
                                     student_name="Rahul Deshmukh"))

    children = face.snapshot().children
    assert children["b"].mood == "speaking"
    assert children["a"].mood != "speaking"


def test_a_snapshot_is_a_copy(bus, face) -> None:
    """The window draws from one of these on another thread; a shared dict
    would be edited underneath it mid-frame."""
    opened(bus)
    bus.publish(ATTENDANCE_MARKED, AttendanceMarked(
        session_id="s1", student_id="a", name="Ananya Sharma", source="roster"))

    look = face.snapshot()
    bus.publish(STUDENT_IDENTIFIED, StudentIdentified(
        student_id="a", track_id=1, source_id="head", zone="front", at=1.0))

    assert look.children["a"].mood == "away", "the drawing changed under the window"
    assert face.snapshot().version > look.version


# --- the window itself ----------------------------------------------------


def built(*extra: str):
    cfg = load("config", "debug", [*HEADLESS, *extra], use_env=False)
    return container.build(cfg, clock=FakeClock(), bus=container.event_bus(cfg))


def test_the_pi_draws_its_own_face() -> None:
    assert load("config", "pi", [], use_env=False).display.face_screen.surface == "pygame"


def test_a_laptop_uses_the_browser_page() -> None:
    assert load("config", "debug", [], use_env=False).display.face_screen.surface == "browser"


def test_the_browser_surface_builds_nothing() -> None:
    system = built("display.face_screen.surface=browser")
    try:
        assert system.face is None, "the /face/ page needs nothing running here"
    finally:
        system.close()


def test_choosing_pygame_builds_it() -> None:
    system = built("display.face_screen.surface=pygame")
    try:
        assert system.face is not None
        assert system.face.name == "pygame"
    finally:
        system.close()


def test_a_face_that_cannot_draw_is_not_a_robot_that_cannot_teach() -> None:
    """run.py starts it through `optional`, the same as the body and the
    camera: a missing pygame is a line in the log."""
    import inspect

    import run

    assert 'optional(logger, "face"' in inspect.getsource(run.main)


def test_the_face_actually_draws(tmp_path) -> None:
    pygame = pytest.importorskip("pygame")
    os.environ["SDL_VIDEODRIVER"] = "dummy"

    pygame.init()
    system = built("display.face_screen.surface=pygame", "display.face_screen.fullscreen=false")
    try:
        surface = pygame.Surface((640, 480))
        # Its own bus: the face only ever subscribes, and a made-up session
        # published onto a running system's bus is an argument with its
        # agents about a school that does not exist.
        own = container.event_bus(system.cfg)
        state = FaceState(own)
        opened(own)
        own.publish(ROBOT_SAY, Utterance(
            text="A leaf needs sunlight, water and a gas from the air.", language="en"))

        drawer = FACE_SURFACES.create("pygame", system.cfg, state)
        big = pygame.font.Font(None, 32)
        drawer._draw(pygame, surface, big, pygame.font.Font(None, 18))

        painted = pygame.image.tostring(surface, "RGB")
        assert len(set(painted)) > 2, "the face drew nothing but its background"
    finally:
        pygame.quit()
        system.close()


def test_a_long_sentence_is_broken_to_fit() -> None:
    pygame = pytest.importorskip("pygame")
    from app.face.pygame_face import wrap

    pygame.font.init()
    font = pygame.font.Font(None, 32)
    lines = wrap("Sunlight from above, water pulled up from the roots, and a gas "
                 "from the air called carbon dioxide.", font, 300)
    assert len(lines) > 1
    assert all(font.size(line)[0] <= 300 for line in lines[:-1])


def test_a_screen_is_found_or_the_reason_is_given() -> None:
    """pygame.init() reports nothing when the video system fails, and the
    first call that needs a screen dies four frames from the cause. On the
    Pi that was `mouse.set_visible` and a traceback at a teacher."""
    pygame = pytest.importorskip("pygame")
    from app.face.pygame_face import open_display

    from lomas_core.schema import ScreenConfig

    pygame.display.quit()
    opened = open_display(pygame, ScreenConfig(driver="dummy"))
    assert opened == "dummy"
    pygame.display.quit()


def test_no_screen_at_all_says_what_to_do() -> None:
    pygame = pytest.importorskip("pygame")
    from lomas_core.errors import LomasError
    from lomas_core.schema import ScreenConfig
    from app.face.pygame_face import open_display

    pygame.display.quit()
    # A driver SDL does not have reads as a pygame with nothing to draw
    # through, which is the more useful of the two things it could mean.
    with pytest.raises(LomasError, match="no way to draw at all|no screen to draw a face on"):
        open_display(pygame, ScreenConfig(driver="not-a-driver"))


def test_the_face_thread_does_not_die_on_a_missing_screen(monkeypatch) -> None:
    """A robot with nowhere to draw still teaches; it says so once."""
    pytest.importorskip("pygame")
    system = built("display.face_screen.surface=pygame", "display.face_screen.driver=not-a-driver")
    try:
        system.face.start()
        system.face._thread.join(timeout=5)
        assert not system.face._thread.is_alive()
    finally:
        system.close()


def test_the_terminals_own_screen_is_tried_first() -> None:
    """A robot started from the desktop it should draw on is the normal
    case, and needs nothing set at all."""
    from lomas_core.schema import ScreenConfig
    from app.face.pygame_face import attempts

    tried = attempts(ScreenConfig())
    assert tried[0] == ("", {})
    assert [driver for driver, _env in tried][-2:] == ["kmsdrm", "fbcon"], "the panel, last"


def test_naming_a_driver_tries_only_that_one() -> None:
    from lomas_core.schema import ScreenConfig
    from app.face.pygame_face import attempts

    assert attempts(ScreenConfig(driver="kmsdrm")) == [("kmsdrm", {})]


def test_a_failed_attempt_leaves_the_environment_as_it_found_it() -> None:
    """Otherwise the second attempt inherits the first one's settings and
    every line of the error says the same thing."""
    import os

    pygame = pytest.importorskip("pygame")
    from lomas_core.errors import LomasError
    from lomas_core.schema import ScreenConfig
    from app.face.pygame_face import open_display

    was = os.environ.get("DISPLAY")
    pygame.display.quit()
    with pytest.raises(LomasError):
        open_display(pygame, ScreenConfig(driver="not-a-driver"))

    assert os.environ.get("DISPLAY") == was
    assert "SDL_VIDEODRIVER" not in os.environ or os.environ["SDL_VIDEODRIVER"] != "not-a-driver"


def test_a_running_desktop_is_found_not_guessed(tmp_path, monkeypatch) -> None:
    """Started over ssh the robot has no screen of its own, and the one
    somebody is looking at is right there, belonging to the same user. Two
    guesses at ":0" and ":1" found neither."""
    from app.face.pygame_face import running_sessions

    (tmp_path / "wayland-0").touch()
    (tmp_path / "wayland-0.lock").touch()
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))

    found = running_sessions()

    waylands = [env for driver, env in found if driver == "wayland"]
    assert [env["WAYLAND_DISPLAY"] for env in waylands] == ["wayland-0"], "the lock is not a screen"


def test_an_x_session_is_tried_with_every_cookie_and_then_none(monkeypatch, tmp_path) -> None:
    """X refuses a client that cannot prove which session it belongs to, and
    where that proof is kept depends on what started the desktop. A wrong
    cookie is also refused where no cookie would have been let in."""
    import os

    from app.face.pygame_face import X_SOCKETS, cookies, running_sessions

    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setenv("XAUTHORITY", str(tmp_path / "cookie"))
    (tmp_path / "cookie").touch()

    assert cookies()[0].endswith("cookie")
    assert cookies()[-1] == "", "no cookie at all is the last thing to try"

    if not os.path.isdir(X_SOCKETS):
        pytest.skip("no X sockets on this machine, which is the point of looking")

    sessions = [env for driver, env in running_sessions() if driver == "x11"]
    assert sessions, "an X socket exists and nothing looked for it"
    assert {env["DISPLAY"] for env in sessions} and all("XAUTHORITY" in env for env in sessions)


def test_a_named_driver_still_wins() -> None:
    from lomas_core.schema import ScreenConfig
    from app.face.pygame_face import attempts

    assert attempts(ScreenConfig(driver="kmsdrm")) == [("kmsdrm", {})]


def test_an_sdl_with_no_backends_is_told_apart_from_a_missing_screen() -> None:
    """The Pi's packaged pygame refused every driver by name while xdpyinfo
    talked to the same display perfectly well. That is a broken pygame, not
    a robot in the wrong terminal, and the advice is different."""
    from lomas_core.errors import LomasError
    from lomas_core.schema import ScreenConfig
    from app.face.pygame_face import open_display

    class NoBackends:
        error = RuntimeError

        class display:
            @staticmethod
            def init():
                raise RuntimeError("x11 not available")

            @staticmethod
            def quit():
                pass

    NoBackends.error = RuntimeError
    with pytest.raises(LomasError, match="no way to draw at all"):
        open_display(NoBackends, ScreenConfig(driver="x11"))


def test_escape_leaves_fullscreen_and_q_closes_the_face() -> None:
    """A face covering the whole screen with no way back is a robot killed
    from another terminal, which is the wrong thing to be doing in front of
    a class."""
    pygame = pytest.importorskip("pygame")

    system = built("display.face_screen.surface=pygame")
    try:
        face = system.face
        modes = []
        pygame.display.set_mode = lambda size, flags=0: modes.append(flags) or "surface"

        face._full = True
        face._keyed(pygame, pygame.K_ESCAPE, "surface")
        assert modes == [0], "escape did not leave fullscreen"

        face._keyed(pygame, pygame.K_ESCAPE, "surface")
        assert modes[-1] == pygame.FULLSCREEN, "and does not go back"

        face._keyed(pygame, pygame.K_q, "surface")
        assert face._stop.is_set()
    finally:
        system.close()
