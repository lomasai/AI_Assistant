from __future__ import annotations

from lomas_llm.provider import PROVIDERS
from lomas_llm.providers.openai_compatible import OpenAiCompatible


@PROVIDERS.register("openai")
class OpenAiProvider(OpenAiCompatible):
    name = "openai"
