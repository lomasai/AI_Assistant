"""Press to talk.

The robot cannot tell which of forty children spoke, and guessing wrong
records one child's answer against another. So attribution is the teacher
tapping a name, and most of what follows is about that not slipping.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from lomas_core.clock import FakeClock
from lomas_core.config import load
from lomas_core.contracts import QUESTION_ANSWERED, QUESTION_ASKED, QUIZ_ANSWERED, ROBOT_STATE
from lomas_core.errors import LomasError
from lomas_speech.recorder import Recorder
from lomas_speech.types import Transcript

from app import container, seed
from app.listener import Listener
from app.web.server import create_app

HEADLESS = [
    "storage.backend=memory",
    "vision.pipeline.enabled=false",
    "hardware.enabled=false",
    "speech.tts.engine=null",
    "speech.stt.engine=keyboard",
    "speech.wake.engine=keyboard",
    "llm.provider=offline",
    "flow.attendance_wait_seconds=1",
    "flow.answer_wait_seconds=1",
    "flow.tick_seconds=0.1",
]

SPOKEN = "why do leaves look green"
MICROPHONE = "microphone"


def spoken_audio(peak: float = 0.4, seconds: float = 1.0, rate: int = 16000) -> bytes:
    """A WAV with real signal in it. The listener measures loudness before it
    spends a network call, so a fixture of zeroes now reads as silence."""
    import math
    import struct

    from lomas_speech.player import wrap_pcm

    level = int(peak * 32767)
    pcm = b"".join(
        struct.pack("<h", int(level * math.sin(2 * math.pi * 220 * i / rate)))
        for i in range(int(seconds * rate))
    )
    return wrap_pcm(pcm, rate)


class FakeMic:
    def __init__(self, audio: bytes | None = None) -> None:
        self.audio = spoken_audio() if audio is None else audio
        self.calls: list[tuple[float, int]] = []
        self.available = True

    def record(self, seconds: float, sample_rate: int) -> bytes:
        self.calls.append((seconds, sample_rate))
        return self.audio

    def describe(self) -> str:
        return "fake"

    def stop(self) -> None: ...


class FakeEars:
    def __init__(self, text: str = SPOKEN) -> None:
        self.text = text
        self.given: list[bytes] = []

    def transcribe(self, audio: bytes, language: str = "") -> Transcript:
        self.given.append(audio)
        return Transcript(text=self.text, language=language or "en")


def build(*extra: str):
    cfg = load("config", "debug", [*HEADLESS, *extra], use_env=False)
    system = container.build(cfg, clock=FakeClock(), bus=container.event_bus(cfg))
    seed.demo_class(system)
    return system


@pytest.fixture
def system():
    built = build()
    yield built
    built.close()


@pytest.fixture
def listening(system):
    system.listener = Listener(system.cfg, system.bus, system.clock, FakeMic(), FakeEars())
    return system.listener


@pytest.fixture
def client(system, listening):
    with TestClient(create_app(system)) as opened:
        yield opened


def seen(system, event: str) -> list:
    return [p for _n, p in system.bus.replay(event)]


# --- the microphone -------------------------------------------------------


def test_no_microphone_is_absence_not_a_crash() -> None:
    """A laptop with no arecord and no portaudio. The robot still teaches; it
    simply cannot be spoken to."""
    silent = Recorder(choice="none")

    assert not silent.available
    assert silent.describe() == "none"
    with pytest.raises(LomasError, match=MICROPHONE):
        silent.record(1.0, 16000)


def test_a_backend_that_is_not_installed_resolves_to_none() -> None:
    """arecord ships with Raspberry Pi OS, which is why the default costs no
    install. Naming one that is absent gives silence, not a crash."""
    assert Recorder(choice="definitely-not-a-real-binary").describe() == "none"


def test_a_named_device_reaches_the_command_line() -> None:
    """A USB microphone is plughw:1,0, not the default input. Getting that
    wrong records silence from the built-in one and looks like a bug
    somewhere else entirely."""
    recorder = Recorder(choice="none", device="plughw:1,0")
    recorder.backend = "arecord"  # the binary need not exist to check the argv

    argv = recorder._argv(Path("out.wav"), 6.0, 16000)

    assert "-D" in argv and "plughw:1,0" in argv
    assert "16000" in argv and "S16_LE" in argv


def test_an_exact_command_overrides_the_guess() -> None:
    recorder = Recorder(choice="none", command="parecord --rate 16000")
    recorder.command = "parecord --rate 16000"

    assert recorder._argv(Path("out.wav"), 6.0, 16000)[:3] == ["parecord", "--rate", "16000"]


# --- what a press produces ------------------------------------------------


def test_a_press_becomes_a_question_with_a_name(system, listening) -> None:
    heard = listening.listen(session_id="s", student_id="s1", student_name="Ananya")

    assert heard["text"] == SPOKEN
    asked = seen(system, QUESTION_ASKED)
    assert len(asked) == 1
    assert asked[0].text == SPOKEN
    assert asked[0].student_id == "s1"
    assert asked[0].student_name == "Ananya"


def test_the_face_shows_that_it_is_hearing_you(system, listening) -> None:
    """The most reassuring screen in the product, and it costs one event."""
    listening.listen(session_id="s", student_id="s1")

    states = [p["state"] for p in seen(system, ROBOT_STATE) if p.get("by") == MICROPHONE]
    assert states == ["listening", "idle"]


def test_the_face_stops_listening_even_when_the_microphone_fails(system) -> None:
    """A device unplugged mid-sentence must not leave the robot staring."""

    class Broken:
        available = True

        def record(self, seconds, sample_rate):
            raise LomasError("the device went away")

        def describe(self) -> str:
            return "broken"

    listener = Listener(system.cfg, system.bus, system.clock, Broken(), FakeEars())
    with pytest.raises(LomasError):
        listener.listen(session_id="s")

    states = [p["state"] for p in seen(system, ROBOT_STATE) if p.get("by") == MICROPHONE]
    assert states[-1] == "idle", "the face was left listening forever"


def test_silence_is_reported_not_published(system, listening) -> None:
    """A child who says nothing must not become an empty question."""
    listening.recorder = FakeMic(audio=b"")
    heard = listening.listen(session_id="s", student_id="s1")

    assert heard["text"] == ""
    assert not seen(system, QUESTION_ASKED)


def test_an_empty_transcript_is_not_a_question(system, listening) -> None:
    listening.stt = FakeEars(text="   ")

    assert listening.listen(session_id="s", student_id="s1")["text"] == ""
    assert not seen(system, QUESTION_ASKED)


def test_the_length_and_rate_come_from_config(system, listening) -> None:
    listening.listen(session_id="s")

    seconds, rate = listening.recorder.calls[0]
    assert seconds == system.cfg.speech.audio.record_seconds
    assert rate == system.cfg.speech.audio.sample_rate == 16000, "whisper wants 16 kHz"


def test_the_tutor_answers_what_was_heard(system, listening) -> None:
    """The whole point: a spoken question is the same event as a typed one,
    so everything downstream already works."""
    ctx = system.orchestrator.open_session()
    listening.listen(session_id=ctx.session_id, student_id="s1", student_name="Ananya")

    answered = seen(system, QUESTION_ANSWERED)
    assert answered and "photosynthesis" in answered[0].answer.lower()


# --- through the teacher screen -------------------------------------------


def test_the_button_uses_the_tapped_student(client, system) -> None:
    ctx = system.orchestrator.open_session()
    student = system.repos["student"].list_for_class(ctx.scope)[1]
    client.post("/api/speaker", json={"student_id": student["id"], "student_name": student["name"]})

    body = client.post("/api/listen", json={}).json()

    assert body["text"] == SPOKEN
    assert seen(system, QUESTION_ASKED)[0].student_id == student["id"]


def test_listening_for_an_answer_records_it_against_the_child(client, system) -> None:
    ctx = system.orchestrator.open_session()
    student = system.repos["student"].list_for_class(ctx.scope)[0]
    client.post("/api/speaker", json={"student_id": student["id"]})

    client.post("/api/listen", json={"as_answer": True})

    answers = seen(system, QUIZ_ANSWERED)
    assert answers and answers[0].student_id == student["id"]
    assert answers[0].response == SPOKEN
    assert answers[0].correct is None, "free text is left for the quizmaster to read"


def test_a_machine_with_no_microphone_says_so(system) -> None:
    """And the button is absent rather than failing when pressed."""
    with TestClient(create_app(system)) as client:
        assert system.listener is None

        refused = client.post("/api/listen", json={})
        assert refused.status_code == 400
        assert MICROPHONE in refused.json()["error"]
        assert client.get("/api/state").json()["microphone"] == "none"


def test_the_face_is_not_told_to_expect_a_wake_phrase(client) -> None:
    """A microphone the teacher presses is not a phrase a child can say, and
    the face must only invite the second when something waits for it."""
    body = client.get("/api/display").json()

    assert body["listening"] is True
    assert body["wake_listening"] is False


def test_nothing_empty_reaches_the_report(client, system) -> None:
    ctx = system.orchestrator.open_session()
    client.post("/api/listen", json={})
    system.orchestrator.close_session()

    report = client.get(f"/api/report/{ctx.session_id}").json()
    for question in report["questions"]:
        assert question["text"], "an empty question reached the report"


# --- silence must not become a sentence -----------------------------------


def test_a_quiet_room_never_reaches_the_model(system, listening) -> None:
    """Found on a Pi: whisper returned "." and "So, let's go." for a room
    that said nothing, and both were published as questions from a named
    child. Loudness is checked before the network, not after."""
    listening.recorder = FakeMic(audio=spoken_audio(peak=0.001))
    ears = listening.stt

    heard = listening.listen(session_id="s", student_id="s1")

    assert heard["text"] == ""
    assert heard["reason"] == "nothing was heard"
    assert not ears.given, "silence was sent to the transcriber anyway"
    assert not seen(system, QUESTION_ASKED)


def test_punctuation_is_not_a_question(system, listening) -> None:
    """What whisper returns for noise. It is not a child asking something."""
    listening.stt = FakeEars(text=".")

    heard = listening.listen(session_id="s", student_id="s1")

    assert heard["text"] == ""
    assert heard["discarded"] == "."
    assert not seen(system, QUESTION_ASKED)


def test_a_real_sentence_still_gets_through(system, listening) -> None:
    """The gate must not be so keen that it eats actual speech."""
    heard = listening.listen(session_id="s", student_id="s1")

    assert heard["text"] == SPOKEN
    assert heard["peak"] > system.cfg.speech.audio.silence_peak
    assert seen(system, QUESTION_ASKED)


def test_the_threshold_is_config(system, listening) -> None:
    quiet = spoken_audio(peak=0.05)
    listening.recorder = FakeMic(audio=quiet)

    system.cfg.speech.audio.silence_peak = 0.5
    assert listening.listen(session_id="s")["text"] == ""

    system.cfg.speech.audio.silence_peak = 0.01
    assert listening.listen(session_id="s")["text"] == SPOKEN
