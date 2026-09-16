"""Everything the first trace from a real Pi found.

Each of these passed the rest of the suite and failed on the robot, which is
the argument for measuring before changing anything. Kept together so the
reason for each is in one place.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from lomas_core.clock import FakeClock
from lomas_core.config import load
from lomas_core.contracts import ROBOT_SAY, ROBOT_SPOKE, Utterance
from lomas_core.errors import LomasError
from lomas_core.schema import FaceConfig
from lomas_face import EMBEDDERS
from lomas_speech.types import SpeechHandle

import app.observability.host as host
from app import container, seed
from app.listener import Listener
from app.voice import Voice
from app.web.server import create_app

HEADLESS = [
    "storage.backend=memory",
    "hardware.enabled=false",
    "vision.pipeline.enabled=false",
    "speech.tts.engine=null",
    "speech.stt.engine=keyboard",
    "speech.wake.engine=keyboard",
    "llm.provider=offline",
]


def build(*extra: str):
    cfg = load("config", "debug", [*HEADLESS, *extra], use_env=False)
    system = container.build(cfg, clock=FakeClock(), bus=container.event_bus(cfg))
    seed.demo_class(system)
    return system


# --- the process table was empty in all 703 samples -----------------------


def fake_proc(root: Path, pid: int, name: str, jiffies: int, pages: int) -> None:
    folder = root / str(pid)
    folder.mkdir(exist_ok=True)
    fields = ["S"] + ["0"] * 10 + [str(jiffies), "0"] + ["0"] * 30
    (folder / "stat").write_text(f"{pid} ({name}) " + " ".join(fields))
    (folder / "statm").write_text(f"1000 {pages} 0 0 0 0 0")


def test_the_process_table_is_not_empty_after_the_first_sample(tmp_path, monkeypatch) -> None:
    """Pruning by what had been reported threw away every baseline on the
    first call, which has nothing to report yet - so no later call ever had
    one, and the table that was meant to say "it was Chromium" said nothing."""
    monkeypatch.setattr(host, "PROC", tmp_path)
    sampler = host.Host()

    fake_proc(tmp_path, 100, "chromium", 1000, 25600)
    fake_proc(tmp_path, 200, "python", 500, 12800)
    assert sampler.processes(10) == [], "nothing to compare against yet"

    time.sleep(0.2)
    fake_proc(tmp_path, 100, "chromium", 1020, 25600)
    fake_proc(tmp_path, 200, "python", 504, 12800)
    second = sampler.processes(10)

    assert [p["name"] for p in second] == ["chromium", "python"], "busiest first"
    assert second[0]["cpu"] > second[1]["cpu"]
    assert second[0]["rss_mb"] == 100.0


def test_a_process_that_exits_is_forgotten(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(host, "PROC", tmp_path)
    sampler = host.Host()

    fake_proc(tmp_path, 300, "arecord", 10, 100)
    sampler.processes(10)

    import shutil

    shutil.rmtree(tmp_path / "300")
    sampler.processes(10)
    assert 300 not in sampler._process_time


# --- two voices at once, and a sixteen second button ----------------------


class Slow:
    """A speaker that takes a known time, and notices if it is used twice at
    once - which is what the Pi did."""

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        self.busy = False
        self.overlapped = False
        self.spoken: list[str] = []
        self._lock = threading.Lock()

    def speak(self, text: str, language: str = "") -> SpeechHandle:
        with self._lock:
            if self.busy:
                self.overlapped = True
            self.busy = True
        time.sleep(self.seconds)
        with self._lock:
            self.busy = False
        self.spoken.append(text)
        handle = SpeechHandle(text=text, language=language)
        handle.finish()
        return handle

    def stop(self) -> None: ...

    def amplitude(self) -> float:
        return 0.0


@pytest.fixture
def voice():
    system = build()
    speaker = Slow(seconds=0.15)
    system.voice.stop()
    system.voice = Voice(speaker, system.extras["gate"], system.bus, wait_seconds=5.0)
    yield system, speaker
    system.voice.stop()
    system.close()


def test_the_robot_never_speaks_over_itself(voice) -> None:
    """The quiz asked its next question while the tutor was still answering,
    from two threads, through one audio process."""
    system, speaker = voice

    def lesson() -> None:
        system.bus.publish(ROBOT_SAY, Utterance(text="the next question", language="en"))

    def tutor() -> None:
        system.bus.publish(ROBOT_SAY, Utterance(text="an answer", language="en", blocking=False))

    threads = [threading.Thread(target=f) for f in (tutor, lesson, tutor, lesson)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    deadline = time.monotonic() + 5
    while len(speaker.spoken) < 4 and time.monotonic() < deadline:
        time.sleep(0.02)

    assert len(speaker.spoken) == 4
    assert not speaker.overlapped, "two sentences were spoken at the same time"


def test_an_agent_does_not_wait_for_its_answer_to_be_read_aloud(voice) -> None:
    """The Ask button held the browser for sixteen seconds - the length of the
    answer - because whoever published the answer waited to hear it."""
    system, speaker = voice
    speaker.seconds = 1.0

    started = time.monotonic()
    system.bus.publish(ROBOT_SAY, Utterance(text="a long answer", language="en", blocking=False))
    returned = time.monotonic() - started

    assert returned < 0.2, f"the caller waited {returned:.2f}s"


def test_a_lesson_still_waits_for_each_sentence(voice) -> None:
    """The other half. A lesson that did not wait would say all six segments
    in the time it takes to read the first."""
    system, speaker = voice

    started = time.monotonic()
    system.bus.publish(ROBOT_SAY, Utterance(text="a segment", language="en"))
    waited = time.monotonic() - started

    assert waited >= speaker.seconds * 0.9
    assert speaker.spoken == ["a segment"]
    assert [p.text for _n, p in system.bus.replay(ROBOT_SPOKE)] == ["a segment"]


def test_utterances_block_unless_they_say_otherwise() -> None:
    assert Utterance(text="x", language="en").blocking is True


def test_agents_speak_without_blocking() -> None:
    """Checked on the agent itself, so a new agent inherits it."""
    system = build()
    try:
        ctx = system.orchestrator.open_session()
        tutor = next(a for a in system.agents.agents if a.name == "tutor")
        assembled = system.agents.assembler.for_agent("tutor", ctx.scope, ctx.session_id)
        tutor.say(assembled, "an answer")

        said = [p for _n, p in system.bus.replay(ROBOT_SAY) if p.reason == "tutor"]
        assert said and said[-1].blocking is False
    finally:
        system.close()


# --- the trace middleware copied every video frame ------------------------


def test_http_timing_does_not_wrap_the_video_stream() -> None:
    """@app.middleware("http") piped every body chunk through a second
    stream, so each MJPEG frame paid an extra copy to be measured, and on
    Ctrl-C it was that stream that raised."""
    system = build("trace.enabled=true", "trace.sample_seconds=0",
                   "trace.directory=data/test-traces")
    try:
        spans: list[dict] = []
        system.trace.span = lambda name, seconds, detail=None: spans.append(detail or {})

        with TestClient(create_app(system)) as client:
            client.get("/api/state")
            with client.stream("GET", "/camera.mjpeg") as response:
                assert response.status_code == 200

        paths = [s.get("path") for s in spans]
        assert "/api/state" in paths
        assert "/camera.mjpeg" not in paths, "an endless stream has no duration"
        assert all(s.get("status") for s in spans)
    finally:
        system.close()
        import shutil

        shutil.rmtree("data/test-traces", ignore_errors=True)


def test_the_timing_is_plain_asgi() -> None:
    import inspect

    from app.web import server

    source = inspect.getsource(server.create_app)
    assert '@app.middleware("http")' not in source


# --- "recognition off" on every run ----------------------------------------


def test_the_default_embedder_needs_nothing_extra_installed() -> None:
    """arcface_onnx needed onnxruntime and a model with no obvious home, so
    the Pi said "recognition off" on every run. SFace runs on the OpenCV
    already there for the detector."""
    cfg = load("config", "pi", [], use_env=False)

    assert cfg.face.embedder == "sface"
    assert "sface" in EMBEDDERS.keys()
    assert cfg.face.embedding_dim == 128


def test_a_missing_model_says_how_to_get_it(tmp_path) -> None:
    embedder = EMBEDDERS.create("sface", FaceConfig(embedder_model_path=str(tmp_path / "gone.onnx")))

    with pytest.raises(LomasError, match="fetch_models"):
        embedder.embed(np.zeros((120, 120, 3), dtype=np.uint8))


def test_the_sface_model_embeds_when_it_is_there() -> None:
    path = Path("models/face_recognition_sface_2021dec.onnx")
    if not path.exists():
        pytest.skip("run python tools/fetch_models.py to test against the real model")

    embedder = EMBEDDERS.create("sface", FaceConfig(embedder_model_path=str(path)))
    rng = np.random.default_rng(3)
    face = (rng.random((160, 160, 3)) * 255).astype(np.uint8)

    vector = embedder.embed(face)
    assert vector.shape == (128,)
    assert abs(float(np.linalg.norm(vector)) - 1.0) < 1e-3
    assert float(1 - np.dot(vector, embedder.embed(face))) < 1e-4


# --- the fetch tool refuses a pointer saved as a model --------------------


def test_a_git_lfs_pointer_is_not_mistaken_for_a_model(tmp_path, monkeypatch) -> None:
    """The first YuNet download was 131 bytes: an LFS pointer with a model's
    name, which loads as garbage and fails somewhere far from the download."""
    import importlib.util

    import sys

    spec = importlib.util.spec_from_file_location("fetch_models", "tools/fetch_models.py")
    fetch = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "fetch_models", fetch)  # dataclasses look themselves up
    spec.loader.exec_module(fetch)
    monkeypatch.setattr(fetch, "MODELS", tmp_path)

    model = fetch.Model("thing.onnx", "http://unused", 200_000, "a test")
    (tmp_path / "thing.onnx").write_text("version https://git-lfs.github.com/spec/v1\n")
    assert not fetch.present(model)

    (tmp_path / "thing.onnx").write_bytes(b"\0" * 250_000)
    assert fetch.present(model)


# --- the second run: mute, with a traceback per sentence ------------------


def test_the_speaker_is_named_not_numbered() -> None:
    """Card numbers are handed out at boot. After a reboot plughw:0,0 was an
    HDMI port, aplay said "Unknown error 524", and the robot taught mute."""
    cfg = load("config", "pi", [], use_env=False)

    assert "CARD=" in cfg.speech.tts.player_device


def test_a_playback_failure_reaches_the_voice_once(caplog) -> None:
    """piper plays on its own thread, so the failure was raised where nobody
    could catch it: a traceback per sentence, and no "no voice" line."""
    from lomas_speech.types import SpeechHandle

    class Failing:
        def speak(self, text: str, language: str = "") -> SpeechHandle:
            handle = SpeechHandle(text=text, language=language)
            threading.Thread(target=handle.fail, args=("aplay failed (1): error 524",)).start()
            return handle

        def stop(self) -> None: ...

        def amplitude(self) -> float:
            return 0.0

    system = build()
    system.voice.stop()
    system.voice = Voice(Failing(), system.extras["gate"], system.bus, wait_seconds=5.0)
    try:
        with caplog.at_level("ERROR"):
            for text in ("one", "two", "three"):
                system.bus.publish(ROBOT_SAY, Utterance(text=text, language="en"))

        reported = [r for r in caplog.records if "no voice" in r.getMessage()]
        assert len(reported) == 1
        assert "524" in reported[0].getMessage()
        assert len(system.bus.replay(ROBOT_SPOKE)) == 3, "the lesson carries on"
    finally:
        system.voice.stop()
        system.close()


def test_piper_puts_its_failure_on_the_handle(tmp_path, monkeypatch) -> None:
    from lomas_core.schema import TtsConfig
    from lomas_speech.ttss.piper import PiperTts

    tts = PiperTts(TtsConfig(engine="piper", player="none"))
    handle = SpeechHandle(text="x", language="en")

    class Process:
        def communicate(self, _data):
            return b"\0\0" * 100, b""

    def refuse(_raw, _rate):
        raise LomasError("aplay failed (1): Unknown error 524")

    tts._process = Process()
    monkeypatch.setattr(tts.player, "play_pcm", refuse)
    tts._run("x", handle)

    assert handle.done and "524" in handle.error


# --- fourteen seconds of silence before every paragraph -------------------


class Chunk:
    def __init__(self, text: str) -> None:
        self.audio_int16_bytes = text.encode()
        self.sample_rate = 22050


class FakeVoice:
    """Yields a sentence at a time, like piper, slowly enough to race."""

    def __init__(self, delay: float = 0.0, fail_after: int = -1) -> None:
        self.delay = delay
        self.fail_after = fail_after

    def synthesize(self, text: str):
        for index, sentence in enumerate(s for s in text.split(". ") if s):
            if index == self.fail_after:
                raise RuntimeError("onnx went away")
            time.sleep(self.delay)
            yield Chunk(sentence)


def in_process(voice: FakeVoice, tmp_path):
    from lomas_core.schema import TtsConfig
    from lomas_speech.ttss.piper_python import PiperPythonTts

    model = tmp_path / "en_US-lessac-medium.onnx"
    model.write_bytes(b"")
    tts = PiperPythonTts(TtsConfig(engine="piper_python", player="none", preload=False,
                                   model_dir=str(tmp_path)))
    tts._voices[model] = voice
    played: list[tuple[str, float]] = []
    tts.player.play_pcm = lambda raw, rate: (played.append((raw.decode(), time.monotonic())),
                                             time.sleep(0.05))
    return tts, played


def test_the_pi_speaks_with_the_voice_already_loaded() -> None:
    cfg = load("config", "pi", [], use_env=False)
    assert cfg.speech.tts.engine == "piper_python"
    assert cfg.speech.tts.preload


def test_the_first_sentence_plays_before_the_last_is_made(tmp_path) -> None:
    """The piper command synthesised a whole paragraph before any of it was
    heard. A sentence at a time, the class hears the first almost at once."""
    tts, played = in_process(FakeVoice(delay=0.1), tmp_path)

    started = time.monotonic()
    handle = tts.speak("One. Two. Three. Four", "en")
    assert handle.wait(5)

    assert [text for text, _ in played] == ["One", "Two", "Three", "Four"]
    assert played[0][1] - started < 0.3, "the first sentence waited for the rest"
    assert not handle.error


def test_pausing_stops_the_sentences_still_to_come(tmp_path) -> None:
    tts, played = in_process(FakeVoice(delay=0.1), tmp_path)

    handle = tts.speak("One. Two. Three. Four. Five. Six", "en")
    time.sleep(0.15)
    tts.stop()
    assert handle.cancelled
    time.sleep(0.8)

    assert len(played) < 6
    assert tts.amplitude() == 0.0


def test_a_synthesis_failure_reaches_the_handle(tmp_path) -> None:
    tts, played = in_process(FakeVoice(fail_after=1), tmp_path)

    handle = tts.speak("One. Two. Three", "en")
    assert handle.wait(5)

    assert [text for text, _ in played] == ["One"]
    assert "onnx went away" in handle.error


def test_without_piper_installed_it_says_what_to_install(tmp_path, monkeypatch) -> None:
    import sys

    from lomas_core.schema import TtsConfig
    from lomas_speech.ttss.piper_python import PiperPythonTts

    (tmp_path / "en_US-lessac-medium.onnx").write_bytes(b"")
    monkeypatch.setitem(sys.modules, "piper", None)
    tts = PiperPythonTts(TtsConfig(engine="piper_python", player="none", preload=False,
                                   model_dir=str(tmp_path)))

    with pytest.raises(LomasError, match="pip install piper-tts"):
        tts.speak("hello", "en")


def test_the_real_voice_speaks_a_sentence_at_a_time(monkeypatch) -> None:
    pytest.importorskip("piper")
    if not Path("models/piper/en_US-lessac-medium.onnx").exists():
        pytest.skip("run python tools/fetch_models.py to test against the real voice")

    from lomas_core.schema import TtsConfig
    from lomas_speech.ttss.piper_python import PiperPythonTts

    tts = PiperPythonTts(TtsConfig(engine="piper_python", player="none", preload=False))
    rates: list[int] = []
    monkeypatch.setattr(tts.player, "play_pcm", lambda raw, rate: rates.append(rate) if raw else None)

    handle = tts.speak("A leaf needs three things. Sunlight, water, and air.", "en")
    assert handle.wait(60)

    assert not handle.error
    assert rates == [22050, 22050]


# --- the third run: a quiz that did not wait, and a mic that heard itself --


def test_model_typography_becomes_plain_text() -> None:
    """The Pi console printed "Don?t", "CO?" and "carbon?dioxide"."""
    from lomas_llm.plain import plain_text

    said = "Leaves don\u2019t change water.  \nThey use CO\u2082 and carbon\u2011dioxide \u2014 **really**."
    assert plain_text(said) == "Leaves don't change water. They use CO2 and carbon-dioxide - really."
    assert plain_text("पत्ती हरी होती है") == "पत्ती हरी होती है", "other scripts pass through"


def test_what_an_agent_says_is_already_plain() -> None:
    system = build()
    try:
        ctx = system.orchestrator.open_session()
        tutor = next(a for a in system.agents.agents if a.name == "tutor")
        from lomas_core.contracts import QUESTION_ASKED
        from lomas_llm import Completion

        tutor.deps.llm.complete = lambda *_a, **_k: Completion(text="It\u2019s CO\u2082.", provider="fake")
        assembled = system.agents.assembler.for_agent("tutor", ctx.scope, ctx.session_id)
        system.orchestrator.ctx = ctx
        from lomas_core.contracts import QuestionAsked

        tutor.handle(QUESTION_ASKED, QuestionAsked(session_id=ctx.session_id, text="x"), assembled)
        said = [p.text for _n, p in system.bus.replay(ROBOT_SAY) if p.reason == "tutor"]
        assert said == ["It's CO2."]
    finally:
        system.close()


def pcm(peaks: list[float], chunk_ms: int = 100, rate: int = 16000) -> bytes:
    import struct

    per_chunk = rate * chunk_ms // 1000
    return b"".join(struct.pack("<h", int(p * 32767)) * per_chunk for p in peaks)


def reader(data: bytes):
    import io

    return io.BytesIO(data).read


def endpoint(**given):
    from lomas_speech.recorder import Endpoint

    return Endpoint(**{"level": 0.02, "silence_ms": 300, "no_speech_seconds": 1.0,
                       "chunk_ms": 100, **given})


def test_recording_stops_at_the_pause_after_speaking() -> None:
    """Six fixed seconds cut one child off and made every short answer wait."""
    from lomas_speech.recorder import read_until_quiet

    talk = pcm([0.0, 0.0, 0.4, 0.4, 0.4, 0.0, 0.0, 0.0, 0.0] + [0.4] * 50)
    got = read_until_quiet(reader(talk), 16000, 15.0, endpoint())

    assert len(got) == len(pcm([0.0] * 8)), "stopped three quiet chunks after the voice"


def test_a_child_who_pauses_briefly_is_not_cut_off() -> None:
    from lomas_speech.recorder import read_until_quiet

    talk = pcm([0.4, 0.0, 0.0, 0.4, 0.4, 0.0, 0.0, 0.0])
    got = read_until_quiet(reader(talk), 16000, 15.0, endpoint())

    assert len(got) == len(talk)


def test_nobody_speaking_gives_up_early() -> None:
    from lomas_speech.recorder import read_until_quiet

    got = read_until_quiet(reader(pcm([0.0] * 100)), 16000, 15.0, endpoint(no_speech_seconds=1.0))
    assert len(got) == len(pcm([0.0] * 10))


def test_the_longest_turn_is_still_a_limit() -> None:
    from lomas_speech.recorder import read_until_quiet

    got = read_until_quiet(reader(pcm([0.4] * 100)), 16000, 2.0, endpoint())
    assert len(got) == len(pcm([0.4] * 20))


def test_listening_waits_for_the_robot_to_finish() -> None:
    """Pressed mid-question, the answer came back as "Sunlight What is the
    green colour inside a leaf called?" - the robot's own voice."""
    from tests.test_listener import FakeEars, FakeMic

    class Talking:
        def __init__(self, polls: int) -> None:
            self.polls = polls

        def is_muted(self) -> bool:
            self.polls -= 1
            return self.polls >= 0

    system = build()
    try:
        gate = Talking(polls=5)
        mic = FakeMic()
        mic.record = lambda seconds, rate, endpoint=None: (mic.calls.append(gate.polls), mic.audio)[1]
        listener = Listener(system.cfg, system.bus, system.clock, mic, FakeEars(), gate)

        listener.listen()

        assert mic.calls == [-1], "recording started only once the robot was quiet"
    finally:
        system.close()


def test_a_listen_during_a_quiz_question_is_its_answer() -> None:
    """The tutor answered each quiz answer as if it were a question, and its
    explanation played over the next quiz question."""
    from lomas_core.contracts import QUESTION_ASKED, QUIZ_ANSWERED
    from tests.test_listener import FakeEars, FakeMic

    system = build()
    try:
        system.listener = Listener(system.cfg, system.bus, system.clock, FakeMic(), FakeEars())
        with TestClient(create_app(system)) as client:
            ctx = system.orchestrator.open_session()
            system.orchestrator.ctx = ctx
            student = system.repos["student"].list_for_class(ctx.scope)[0]
            client.post("/api/speaker", json={"student_id": student["id"]})
            ctx.notes["quiz_posed"] = "q3"

            client.post("/api/listen", json={})

            answers = [p for _n, p in system.bus.replay(QUIZ_ANSWERED)]
            assert [a.question_id for a in answers] == ["q3"]
            assert not system.bus.replay(QUESTION_ASKED), "the tutor was not asked"
    finally:
        system.close()


def test_the_answer_wait_starts_once_the_question_is_heard() -> None:
    """q1 was queued behind a nineteen second answer and the wait ran out a
    second and a half after the class heard it."""
    from lomas_core.contracts import ROBOT_SAY
    from app.flow.states import StepResult
    from app.flow.steps.quiz import QuizStep

    system = build("flow.answer_wait_seconds=20")
    try:
        ctx = system.orchestrator.open_session()
        system.bus.subscribe(ROBOT_SAY, lambda *_: system.clock.advance(19))
        step = QuizStep(system.cfg)
        step.enter(ctx)

        step.tick(ctx, system.clock.now())
        assert ctx.notes["quiz_index"] == 1
        system.clock.advance(2)
        step.tick(ctx, system.clock.now())

        assert ctx.notes["quiz_index"] == 1, "moved on before anyone could answer"
        step.exit(ctx)
    finally:
        system.close()


def test_a_late_answer_does_not_end_the_wait_on_the_next_question() -> None:
    from lomas_core.contracts import QUIZ_ANSWERED, QuizAnswered
    from app.flow.steps.quiz import QuizStep

    system = build()
    try:
        ctx = system.orchestrator.open_session()
        step = QuizStep(system.cfg)
        step.enter(ctx)
        student = system.repos["student"].list_for_class(ctx.scope)[0]
        ctx.notes["quiz_posed"] = "q4"

        system.bus.publish(QUIZ_ANSWERED, QuizAnswered(
            session_id=ctx.session_id, question_id="q3", student_id=student["id"],
            response="chlorophyll", correct=None, latency_ms=0))

        assert ctx.notes["quiz_posed"] == "q4"
        step.exit(ctx)
    finally:
        system.close()
