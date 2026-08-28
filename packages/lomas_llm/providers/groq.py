from __future__ import annotations

from lomas_llm.provider import PROVIDERS
from lomas_llm.providers.openai_compatible import OpenAiCompatible


@PROVIDERS.register("groq")
class GroqProvider(OpenAiCompatible):
    """Fast enough for classroom turn-taking, which is the only reason to
    send a child's question off the robot at all."""

    name = "groq"
