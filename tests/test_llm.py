"""Providers, routing and prompts - all of it with no API key anywhere."""
from __future__ import annotations

import pytest

from lomas_core.schema import LlmConfig, RouterConfig
from lomas_llm import PROVIDERS, Complexity, Message, PromptLibrary, Router
from lomas_llm.types import USER

PROMPTS = "config/prompts"
BANNED_IN_NUDGES = ("distracted", "attention", "listening", "concentrate", "focus")


def cfg(**overrides) -> LlmConfig:
    base = dict(provider="offline", prompts_path=PROMPTS, fallback_language="en")
    base.update(overrides)
    return LlmConfig(**base)


@pytest.fixture
def prompts() -> PromptLibrary:
    return PromptLibrary(PROMPTS, "en")


@pytest.fixture
def offline():
    return PROVIDERS.create("offline", cfg())


def ask(text: str) -> list[Message]:
    return [Message(USER, text)]


# --- providers -------------------------------------------------------------


def test_every_provider_is_registered():
    assert set(PROVIDERS.keys()) == {"offline", "groq", "anthropic", "openai"}


def test_cloud_providers_construct_without_keys_or_sdks():
    """They must build on any machine and only complain when actually used."""
    for name in ("groq", "anthropic", "openai"):
        assert PROVIDERS.create(name, cfg(provider=name)).name == name


def test_swapping_provider_changes_only_the_class():
    built = {name: type(PROVIDERS.create(name, cfg(provider=name))).__name__
             for name in PROVIDERS.keys()}
    assert len(set(built.values())) == len(built)


def test_offline_answers_from_the_faq(offline):
    answer = offline.complete(ask("why are leaves green")).text
    assert "photosynthesis" in answer.lower()
    assert offline.complete(ask("tell me about rain and clouds")).text


def test_offline_falls_back_politely(offline):
    answer = offline.complete(ask("what is the capital of Peru")).text
    assert answer, "must always say something"
    assert "connected" in answer.lower()


def test_offline_streams_the_same_text(offline):
    question = ask("what is photosynthesis")
    assert "".join(offline.stream(question)) == offline.complete(question).text


def test_offline_reports_usage(offline):
    assert offline.complete(ask("why are leaves green")).usage.output_tokens > 0


def test_empty_completion_is_falsy():
    from lomas_llm.types import Completion

    assert not Completion(text="   ", provider="x")


# --- routing ---------------------------------------------------------------


def router(**overrides) -> Router:
    return Router(RouterConfig(**overrides))


def test_a_plain_question_is_simple():
    assert router().classify("what is a leaf") is Complexity.SIMPLE


def test_reasoning_words_raise_the_score():
    decision = router().route("why do leaves look green")
    assert "reasoning" in decision.reasons


def test_a_long_multipart_reasoning_question_is_complex():
    question = (
        "why do leaves look green and how does the plant actually turn sunlight "
        "into food, and also what happens to the plant during the night when "
        "there is no sunlight at all to work with"
    )
    decision = router().route(question)
    assert decision.complexity is Complexity.COMPLEX
    assert decision.provider == "anthropic"
    assert set(decision.reasons) == {"long", "reasoning", "multipart"}


def test_routing_picks_exactly_one_provider():
    for question in ("what is a leaf", "why is the sky blue", "explain and compare X and Y?"):
        assert isinstance(router().route(question).provider, str)


def test_thresholds_are_config():
    strict = router(medium_at=1, complex_at=2)
    assert strict.classify("why do leaves look green") is Complexity.MEDIUM


def test_router_can_be_switched_off():
    assert router(enabled=False).classify("why and how and explain?") is Complexity.SIMPLE


def test_provider_names_come_from_config():
    picked = router(simple_provider="offline").route("what is a leaf")
    assert picked.provider == "offline"


# --- prompts ---------------------------------------------------------------


def test_prompt_library_finds_the_english_set(prompts):
    assert {"tutor", "quizmaster", "narrator", "safety", "nudge", "offline"} <= set(
        prompts.available("en")
    )


def test_tutor_prompt_renders_with_its_variables(prompts):
    messages = prompts.messages(
        "tutor", "en", grade="Grade 6", subject="science", vocabulary_level="middle",
        language="English", topic="photosynthesis", student_name="Ananya",
        question="why are leaves green",
    )
    assert [m.role for m in messages] == ["system", "user"]
    assert "Ananya" in messages[1].content
    assert "Grade 6" in messages[0].content


def test_a_missing_variable_names_itself(prompts):
    with pytest.raises(Exception, match="student_name"):
        prompts.messages("tutor", "en", grade="6", subject="science",
                         vocabulary_level="middle", language="English",
                         topic="x", question="y")


def test_hindi_uses_its_own_file_not_a_translation_of_english(prompts):
    english = prompts.line("nudge", "en", chooser=lambda o: o[0], name="Meera")
    hindi = prompts.line("nudge", "hi", chooser=lambda o: o[0], name="Meera")
    assert english != hindi
    assert "Meera" in hindi


def test_a_language_without_a_file_falls_back(prompts):
    """Marathi has no prompts yet; the robot must still work."""
    assert prompts.line("nudge", "mr", chooser=lambda o: o[0], name="Kabir")


def test_unknown_prompt_lists_what_exists(prompts):
    with pytest.raises(Exception, match="tutor"):
        prompts.messages("does_not_exist", "en")


def test_every_nudge_is_an_invitation(prompts):
    """A robot that tells a child off in front of the class gets switched off
    and never switched on again."""
    for language in ("en", "hi"):
        block, _ = prompts._load("nudge", language)
        for line in block["lines"]:
            lowered = line.lower()
            for banned in BANNED_IN_NUDGES:
                assert banned not in lowered, f"{language}: '{line}' reprimands"
            assert "?" in line or "{name}" in line


def test_nudges_vary(prompts):
    block, _ = prompts._load("nudge", "en")
    assert len(block["lines"]) > 2, "the robot must not repeat itself all lesson"
