from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import yaml

from lomas_core.errors import LomasError
from lomas_llm.types import Message, SYSTEM, USER

SYSTEM_KEY = "system"
USER_KEY = "user"
LINES_KEY = "lines"


class PromptLibrary:
    """Every prompt the system uses, keyed by language.

    `prompt` and `language` are positional-only so that `name` and `language`
    stay free as template variables - a tutor prompt wants a student name and
    a language to answer in, and both would otherwise collide with the
    selector arguments.

    No prompt text may live in a .py file. The moment an English sentence is
    inlined into code, adding Hindi becomes a code change instead of a content
    change, and the multilingual promise quietly dies. A test enforces this.
    """

    def __init__(self, root: str | Path, fallback_language: str) -> None:
        self.root = Path(root)
        self.fallback_language = fallback_language
        self._cache: dict[tuple[str, str], dict[str, Any]] = {}

    def languages(self) -> list[str]:
        if not self.root.is_dir():
            return []
        return sorted(p.name for p in self.root.iterdir() if p.is_dir())

    def available(self, language: str) -> list[str]:
        folder = self.root / language
        if not folder.is_dir():
            return []
        return sorted(p.stem for p in folder.glob("*.yaml"))

    def messages(self, prompt: str, language: str, /, **values: Any) -> list[Message]:
        block, used = self._load(prompt, language)
        if SYSTEM_KEY not in block and USER_KEY not in block:
            raise LomasError(f"prompt '{prompt}' ({used}) has no {SYSTEM_KEY} or {USER_KEY} section")

        built: list[Message] = []
        if block.get(SYSTEM_KEY):
            built.append(Message(SYSTEM, self._fill(block[SYSTEM_KEY], prompt, values)))
        if block.get(USER_KEY):
            built.append(Message(USER, self._fill(block[USER_KEY], prompt, values)))
        return built

    def line(self, prompt: str, language: str, /, *, chooser=random.choice, **values: Any) -> str:
        """One phrasing from a list. Used where the robot should not sound
        identical every time, such as inviting a drifting child back in."""
        block, used = self._load(prompt, language)
        options = block.get(LINES_KEY)
        if not options:
            raise LomasError(f"prompt '{prompt}' ({used}) has no {LINES_KEY} section")
        return self._fill(chooser(options), prompt, values)

    def _load(self, prompt: str, language: str) -> tuple[dict[str, Any], str]:
        for candidate in (language, self.fallback_language):
            key = (candidate, prompt)
            if key in self._cache:
                return self._cache[key], candidate

            path = self.root / candidate / f"{prompt}.yaml"
            if not path.exists():
                continue
            loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if not isinstance(loaded, dict):
                raise LomasError(f"{path}: expected a mapping")
            self._cache[key] = loaded
            return loaded, candidate

        known = ", ".join(self.available(language)) or "none"
        raise LomasError(
            f"no prompt '{prompt}' for '{language}' or fallback "
            f"'{self.fallback_language}'. Available in '{language}': {known}"
        )

    def _fill(self, template: str, prompt: str, values: dict[str, Any]) -> str:
        try:
            return template.format(**values).strip()
        except KeyError as exc:
            raise LomasError(f"prompt '{prompt}' needs a value for {exc}") from None
