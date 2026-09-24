"""A knob for the room.

The same robot is too quiet for forty children after lunch and too loud for
six of them in a library, and the teacher is the one who knows which. So: a
slider on the page, a control that moves the card's own mixer where there is
one, and a level that survives being switched off at the wall.
"""
from __future__ import annotations

import array
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from lomas_core.clock import FakeClock
from lomas_core.config import load
from lomas_core.schema import VolumeConfig
from lomas_speech.player import Player, scale_pcm
from lomas_speech.volume import volume_control
from lomas_speech.volumes.alsa import _card_of

from app import container, seed
from app.web.server import create_app

HEADLESS = [
    "storage.backend=memory",
    "vision.pipeline.enabled=false",
    "speech.stt.engine=keyboard",
    "speech.wake.engine=keyboard",
    "llm.provider=offline",
    # A real engine, because a null voice has no speaker and therefore no
    # knob - but told to play nothing, because this is a test machine.
    "speech.tts.engine=piper",
    "speech.tts.player=none",
    "speech.tts.volume.control=software",
]

LOUD_SAMPLE = 10000
SAMPLES = 100


def sound() -> bytes:
    return array.array("h", [LOUD_SAMPLE] * SAMPLES).tobytes()


def dial(tmp_path, **values):
    return volume_control(VolumeConfig(state_file=str(tmp_path / "volume.json"), **values))


@pytest.fixture
def knob(tmp_path):
    return dial(tmp_path, control="software", level=0.8)


# --- the control ----------------------------------------------------------


def test_the_slider_moves_the_sound(knob) -> None:
    knob.set(0.2)
    was = knob.gain

    knob.set(0.9)

    assert knob.gain > was, "turning it up made no difference to the audio"


def test_halfway_sounds_like_halfway(knob) -> None:
    """Loudness is not linear. Samples at half scale sound much louder than
    half, which is why the level is curved before it reaches them."""
    knob.set(0.5)

    assert knob.gain < 0.5


def test_a_school_can_cap_how_loud_it_gets(tmp_path) -> None:
    capped = dial(tmp_path, control="software", max_level=0.6, min_level=0.1)

    assert capped.set(1.0) == 0.6, "the teacher went past the school's ceiling"
    assert capped.set(0.0) == 0.1


def test_mute_is_silence_and_not_a_whisper(knob) -> None:
    knob.mute(True)

    assert knob.gain == 0.0
    assert knob.level == 0.8, "unmuting must come back to where the slider was"

    knob.mute(False)
    assert knob.gain > 0.0


def test_the_level_survives_being_switched_off(tmp_path) -> None:
    """A classroom robot is turned off at the wall, so the volume the room
    settled on has to be on disk and not in memory."""
    dial(tmp_path, control="software").set(0.35)

    again = dial(tmp_path, control="software")

    assert again.level == 0.35
    assert json.loads((tmp_path / "volume.json").read_text())["level"] == 0.35


def test_a_robot_that_may_not_remember(tmp_path) -> None:
    dial(tmp_path, control="software", remember=False).set(0.3)

    assert not (tmp_path / "volume.json").exists()


def test_nudging_by_a_step(knob) -> None:
    knob.set(0.5)

    assert knob.nudge(1) == pytest.approx(0.5 + knob.cfg.step)
    assert knob.nudge(-2) == pytest.approx(0.5 - knob.cfg.step)


def test_the_control_is_a_config_choice(tmp_path) -> None:
    """Rule 1: how the robot changes its loudness is a config change, and
    a robot wired to an amplifier with its own dial has none."""
    fixed = dial(tmp_path, control="none")

    assert fixed.name == "none"
    assert fixed.set(0.1) == 0.1, "the level is still remembered for the next control"


# --- what reaches the speaker ---------------------------------------------


def test_samples_are_scaled_not_wrapped() -> None:
    """A sample past the top that wraps round is a crack in the middle of a
    word, which is worse than the loudness it was fixing."""
    quieter = array.array("h")
    quieter.frombytes(scale_pcm(sound(), 0.5))

    assert max(quieter) == LOUD_SAMPLE // 2
    assert all(-32768 <= sample <= 32767 for sample in quieter)


def test_full_volume_touches_nothing() -> None:
    raw = sound()

    assert scale_pcm(raw, 1.0) is raw, "every sample was copied for no reason"


def test_a_muted_robot_does_not_wait_for_the_card(tmp_path) -> None:
    """Played at zero, a card still opens and still takes the length of the
    sentence. The lesson would pause for speech nobody can hear."""
    player = Player(choice="none",
                    volume=VolumeConfig(control="software",
                                        state_file=str(tmp_path / "volume.json")))
    player.volume.mute(True)
    played: list[Path] = []
    player._play_command = lambda path, backend: played.append(path)

    player.play_file(Path("nothing.wav"))

    assert played == []


def test_the_player_says_how_loud_it_is(tmp_path) -> None:
    player = Player(choice="none",
                    volume=VolumeConfig(control="software", level=0.4,
                                        state_file=str(tmp_path / "volume.json")))

    assert "40%" in player.describe()


# --- the card's own mixer -------------------------------------------------


def test_the_mixer_card_comes_from_the_speaker() -> None:
    """The card with the speaker on it is already in config as the player
    device, and configuring the same card twice is how they drift apart."""
    assert _card_of("plughw:CARD=Headphones,DEV=0") == "Headphones"
    assert _card_of("default") == ""
    assert _card_of("") == ""


# The Pi's headphone jack, word for word from `amixer -c Headphones sget PCM`.
PI_JACK = """Simple mixer control 'PCM',0
  Capabilities: pvolume pvolume-joined pswitch pswitch-joined
  Playback channels: Mono
  Limits: Playback -10239 - 400
  Mono: Playback -1728 [80%] [-17.28dB] [on]"""

# A card with a volume but no decibel scale, which some USB speakers are.
NO_DECIBELS = """Simple mixer control 'Speaker',0
  Capabilities: pvolume
  Limits: Playback 0 - 65536
  Mono: Playback 32768 [50%] [on]"""


def a_card(tmp_path, reply: str, **values):
    """An ALSA control talking to a card that answers like this."""
    from lomas_speech.volumes.alsa import AlsaVolume

    asked: list[list[str]] = []

    class Card(AlsaVolume):
        def _amixer(self, *args):
            asked.append(list(args))
            return True, reply

    control = Card(VolumeConfig(control="alsa", state_file=str(tmp_path / "v.json"), **values))
    return control, asked


def _sset(asked: list[list[str]]) -> list[str]:
    """The last thing the control asked the mixer to do."""
    return [args for args in asked if "sset" in args][-1]


def _value(asked: list[list[str]]) -> str:
    """The level out of that command, wherever in it the level sits."""
    return next(arg for arg in _sset(asked) if arg.endswith(("dB", "%")))


def test_a_negative_decibel_reaches_the_card(tmp_path) -> None:
    """The bug that made every button useless: amixer reads a leading minus
    as an option, so `sset PCM -4.00dB` failed and only the loudest setting -
    the one with no minus sign - ever worked. Silent at 0, unchanged at 45,
    slightly louder at 100, and the log reporting levels it never applied."""
    _control, asked = a_card(tmp_path, PI_JACK, level=0.8)

    put = _sset(asked)

    assert put[0] == "--", "amixer will read the decibels as a flag"
    assert put.index("--") < put.index("sset")


def test_a_mixer_that_refuses_is_not_kept_quiet(tmp_path, caplog) -> None:
    """A knob that reports a level it never applied argues with the room."""
    import logging

    from lomas_speech.volumes.alsa import AlsaVolume

    class Refuses(AlsaVolume):
        def _amixer(self, *args):
            if "sset" in args:
                return False, "amixer: Unable to find simple control"
            return True, PI_JACK

    with caplog.at_level(logging.WARNING):
        Refuses(VolumeConfig(control="alsa", state_file=str(tmp_path / "v.json"))).set(0.5)

    assert any("refused" in record.getMessage() for record in caplog.records)


def test_the_slider_is_decibels_where_the_card_has_them(tmp_path) -> None:
    """80% of the Pi's jack is -17 dB, which is a seventh of the amplitude
    and inaudible under a fan. A percentage is not a level."""
    control, asked = a_card(tmp_path, PI_JACK, level=0.8, range_db=40.0)

    control.set(0.8)
    put = _value(asked)

    assert put.endswith("dB"), f"it set {put}, which is not a level"
    assert float(put.rstrip("dB")) == pytest.approx(-4.0), "0.8 of a 40 dB span"


def test_the_top_of_the_slider_is_as_loud_as_the_card_goes(tmp_path) -> None:
    control, asked = a_card(tmp_path, PI_JACK, level=1.0)

    control.set(1.0)
    assert float(_value(asked).rstrip("dB")) == pytest.approx(4.0), "the card's own loudest"


def test_the_bottom_is_quiet_rather_than_off(tmp_path) -> None:
    """The card goes down to -102 dB, which is off. A slider whose bottom
    half is silence is a slider with a cliff in it."""
    control, asked = a_card(tmp_path, PI_JACK, level=0.0, range_db=40.0)

    control.set(0.0)
    assert float(_value(asked).rstrip("dB")) == pytest.approx(-36.0)


def test_how_much_range_the_slider_covers_is_config(tmp_path) -> None:
    control, asked = a_card(tmp_path, PI_JACK, level=0.5, range_db=20.0)

    control.set(0.5)
    assert float(_value(asked).rstrip("dB")) == pytest.approx(-6.0)


def test_a_card_with_no_decibel_scale_still_works(tmp_path) -> None:
    control, asked = a_card(tmp_path, NO_DECIBELS, level=0.6)

    control.set(0.6)
    assert _value(asked) == "60%", "a percentage only where there is nothing better"


def test_a_machine_with_no_mixer_still_has_a_slider(tmp_path) -> None:
    """auto prefers the card's own knob and falls back to scaling samples. A
    laptop and a Pi with an HDMI-only card both end up here."""
    chosen = dial(tmp_path, control="auto")

    assert chosen.name in {"alsa", "software"}


def test_the_robot_finds_its_own_knob() -> None:
    """auto rather than alsa: this Pi lists four playback cards and only the
    headphone jack has a mixer. Pinned to alsa and pointed at an i2s DAC,
    the slider would move nothing at all."""
    assert load("config", "pi", [], use_env=False).speech.tts.volume.control == "auto"


# --- the teacher's slider -------------------------------------------------


@pytest.fixture
def client(tmp_path):
    cfg = load("config", "debug",
               [*HEADLESS, f"speech.tts.volume.state_file={tmp_path / 'volume.json'}"],
               use_env=False)
    system = container.build(cfg, clock=FakeClock(), bus=container.event_bus(cfg))
    seed.demo_class(system)
    try:
        with TestClient(create_app(system)) as opened:
            yield opened
    finally:
        system.close()


def test_the_page_can_read_the_volume(client) -> None:
    body = client.get("/api/volume").json()

    assert body["available"] is True
    assert 0 <= body["dial"] <= 100


def test_the_page_can_turn_it_down(client) -> None:
    body = client.post("/api/volume", json={"level": 0.3}).json()

    assert body["dial"] == 30
    assert client.get("/api/volume").json()["dial"] == 30


def test_the_page_can_mute_and_unmute(client) -> None:
    assert client.post("/api/volume", json={"muted": True}).json()["muted"] is True
    assert client.post("/api/volume", json={"muted": False}).json()["muted"] is False


def test_the_level_that_comes_back_is_the_one_that_happened(client) -> None:
    """Not the one asked for: a school can cap the maximum, and some cards
    have only a handful of steps."""
    asked = client.post("/api/volume", json={"level": 5.0}).json()

    assert asked["level"] <= 1.0


def test_the_state_a_page_loads_with_carries_the_volume(client) -> None:
    assert "volume" in client.get("/api/state").json()


def test_a_robot_with_no_speaker_says_so() -> None:
    """A null voice has no player and no knob, and the page hides the slider
    rather than showing one that does nothing."""
    cfg = load("config", "debug",
               ["storage.backend=memory", "vision.pipeline.enabled=false",
                "speech.tts.engine=null", "speech.stt.engine=keyboard",
                "speech.wake.engine=keyboard", "llm.provider=offline"],
               use_env=False)
    system = container.build(cfg, clock=FakeClock(), bus=container.event_bus(cfg))
    try:
        with TestClient(create_app(system)) as opened:
            assert opened.get("/api/volume").json() == {"available": False}
    finally:
        system.close()
