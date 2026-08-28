from __future__ import annotations

import re
from typing import Iterator

from lomas_core.schema import LlmConfig
from lomas_llm.prompts import PromptLibrary
from lomas_llm.provider import PROVIDERS
from lomas_llm.types import SYSTEM, Completion, Message, Usage

WORD = re.compile(r"\w+")
FALLBACK_KEY = "fallback"
NO_OVERLAP = 0


@PROVIDERS.register("offline")
class OfflineProvider:
    """The default, and the one a government school with no internet runs on.

    Answers come from a language-keyed FAQ file, matched by word overlap. It
    is not clever and is not meant to be; it keeps the lesson moving when the
    connection drops, and it means the whole system runs with no API key.
    """

    name = "offline"

    def __init__(self, cfg: LlmConfig, prompts: PromptLibrary | None = None) -> None:
        self.cfg = cfg
        self.prompts = prompts or PromptLibrary(cfg.prompts_path, cfg.fallback_language)
        self.model = cfg.offline_faq

    def complete(self, messages: list[Message], **options) -> Completion:
        language = options.get("language", self.cfg.fallback_language)
        question = self._last_user(messages)
        text = self._answer(question, language)
        return Completion(
            text=text,
            provider=self.name,
            model=self.model,
            usage=Usage(input_tokens=len(WORD.findall(question)),
                        output_tokens=len(WORD.findall(text))),
        )

    def stream(self, messages: list[Message], **options) -> Iterator[str]:
        yield self.complete(messages, **options).text

    def _last_user(self, messages: list[Message]) -> str:
        for message in reversed(messages):
            if message.role != SYSTEM:
                return message.content
        return ""

    def _answer(self, question: str, language: str) -> str:
        entries = self._entries(language)
        asked = set(w.lower() for w in WORD.findall(question))

        best_answer = ""
        best_overlap = NO_OVERLAP
        for keywords, answer in entries:
            overlap = len(asked & keywords)
            if overlap > best_overlap:
                best_overlap = overlap
                best_answer = answer

        if best_overlap > NO_OVERLAP:
            return best_answer
        return self.prompts.line(self.cfg.offline_faq, language, chooser=lambda o: o[0])

    def _entries(self, language: str) -> list[tuple[set[str], str]]:
        block, _ = self.prompts._load(self.cfg.offline_faq, language)
        return [
            (set(w.lower() for w in WORD.findall(item.get("match", ""))), item.get("answer", ""))
            for item in block.get("entries", [])
        ]
