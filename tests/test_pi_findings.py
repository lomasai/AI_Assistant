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
