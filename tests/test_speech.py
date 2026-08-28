"""The whole speech path, with no microphone and no speaker anywhere."""
from __future__ import annotations

import pytest

from lomas_core.clock import FakeClock
from lomas_core.schema import AudioConfig, AudioInputConfig, SttConfig, TtsConfig, WakeConfig
from lomas_speech import (
    STT_ENGINES,
    TTS_ENGINES,
    WAKE_WORDS,
    DuplexGate,
    InputSet,
    Transcript,
)

TAIL_MS = 250
TAIL_SECONDS = TAIL_MS / 1000.0


@pytest.fixture
def wake():
    engine = WAKE_WORDS.create("keyboard", WakeConfig(engine="keyboard"))
    engine.start()
    return engine


@pytest.fixture
def stt():
    return STT_ENGINES.create("keyboard", SttConfig(engine="keyboard", language="en"))


@pytest.fixture
def tts():
    return TTS_ENGINES.create("null", TtsConfig(engine="null"))


def test_every_engine_is_registered():
    assert set(WAKE_WORDS.keys()) == {"openwakeword", "porcupine", "keyboard"}
    assert set(STT_ENGINES.keys()) == {"groq", "vosk", "keyboard"}
    assert set(TTS_ENGINES.keys()) == {"piper", "gtts", "null"}


def test_cloud_engines_import_without_their_dependencies():
    """They must construct on any machine and only complain when used."""
    assert STT_ENGINES.create("groq", SttConfig()) is not None
    assert STT_ENGINES.create("vosk", SttConfig()) is not None
    assert TTS_ENGINES.create("piper", TtsConfig()) is not None


def test_full_cycle_with_no_hardware(wake, stt, tts):
    """Wake, hear, answer - the loop the whole product is built on."""
    assert wake.poll() is None, "silent until triggered"

    wake.trigger(zone="front")
    event = wake.poll()
    assert event is not None
    assert event.phrase == "hey lomas"
    assert event.zone == "front"
    assert wake.poll() is None, "a trigger fires once"

    stt.say("what is photosynthesis")
    heard = stt.transcribe()
    assert heard.text == "what is photosynthesis"
    assert heard.final
    assert heard.language == "en"

    handle = tts.speak("Leaves turn sunlight into food.", "en")
    assert handle.done
    assert tts.last() == "Leaves turn sunlight into food."


def test_wake_carries_the_zone_that_heard_it():
    """Per-desk microphones later need to know which desk spoke."""
    engine = WAKE_WORDS.create("keyboard", WakeConfig(engine="keyboard"))
    engine.start()
    engine.trigger(zone="desk-4")
    assert engine.poll().zone == "desk-4"


def test_stopped_wake_engine_stays_quiet(wake):
    wake.trigger()
    wake.stop()
    assert wake.poll() is None


def test_empty_transcript_is_falsy(stt):
    assert not stt.transcribe()
    stt.say("   ")
    assert not stt.transcribe(), "whitespace is not speech"
    stt.say("hello")
    assert stt.transcribe()


def test_transcript_language_can_be_overridden(stt):
    stt.say("नमस्ते")
    assert stt.transcribe(language="hi").language == "hi"


def test_null_tts_records_language_per_utterance(tts):
    tts.speak("Good morning", "en")
    tts.speak("सुप्रभात", "hi")
    assert tts.spoken == [("Good morning", "en"), ("सुप्रभात", "hi")]


def test_voice_map_is_per_language():
    """One voice string would quietly kill the multilingual promise."""
    voices = TtsConfig().voice
    assert voices["en"] != voices["hi"]


# --- microphones as a list -------------------------------------------------


def test_single_microphone_is_still_a_list():
    inputs = InputSet(AudioConfig())
    assert len(inputs) == 1
    assert inputs.primary.zone == "front"


def test_per_desk_microphones_need_no_code_change():
    """Thirty desks is thirty config entries, which is the whole point."""
    desks = AudioConfig(
        inputs=[AudioInputConfig(id=f"desk-{n}", device=f"hw:{n}", zone=f"row-{n // 5}")
                for n in range(30)]
    )
    inputs = InputSet(desks)

    assert len(inputs) == 30
    assert inputs.zones() == [f"row-{n}" for n in range(6)]
    assert len(inputs.by_zone("row-0")) == 5
    assert inputs.by_id("desk-7").device == "hw:7"


def test_unknown_input_is_named_in_the_error():
    inputs = InputSet(AudioConfig())
    with pytest.raises(Exception, match="desk-99"):
        inputs.by_id("desk-99")


def test_no_microphones_is_refused():
    with pytest.raises(Exception, match="at least one microphone"):
        InputSet(AudioConfig(inputs=[]))


# --- half duplex -----------------------------------------------------------


def test_gate_mutes_while_speaking_and_for_the_tail():
    clock = FakeClock()
    gate = DuplexGate(AudioConfig(half_duplex=True, tail_ms=TAIL_MS), clock)

    assert not gate.is_muted()

    gate.on_speech_start()
    assert gate.is_muted(), "the robot must not hear itself"

    clock.advance(5.0)
    assert gate.is_muted(), "still speaking"

    gate.on_speech_end()
    assert gate.is_muted(), "the room does not fall silent instantly"

    clock.advance(TAIL_SECONDS / 2)
    assert gate.is_muted()

    clock.advance(TAIL_SECONDS)
    assert not gate.is_muted()


def test_gate_can_be_switched_off():
    clock = FakeClock()
    gate = DuplexGate(AudioConfig(half_duplex=False), clock)
    gate.on_speech_start()
    assert not gate.is_muted()


def test_speaking_again_reopens_the_mute():
    clock = FakeClock()
    gate = DuplexGate(AudioConfig(half_duplex=True, tail_ms=TAIL_MS), clock)

    gate.on_speech_start()
    gate.on_speech_end()
    clock.advance(TAIL_SECONDS * 2)
    assert not gate.is_muted()

    gate.on_speech_start()
    assert gate.is_muted()


def test_speech_can_be_cancelled_mid_utterance(tts):
    """The teacher's pause button depends on this."""
    handle = tts.speak("A very long explanation about photosynthesis", "en")
    handle._done.clear()
    tts.stop()
    assert handle.cancelled
    assert handle.done


def test_transcript_is_immutable():
    heard = Transcript(text="hello", language="en")
    with pytest.raises(Exception):
        heard.text = "goodbye"
