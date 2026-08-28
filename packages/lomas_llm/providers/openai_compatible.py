from __future__ import annotations

import json
import os
from typing import Iterator

from lomas_core.errors import LomasError
from lomas_core.schema import LlmConfig
from lomas_llm.provider import LLMProvider
from lomas_llm.types import Completion, Message, Usage

DATA_PREFIX = "data: "
DONE = "[DONE]"
FIRST = 0


class OpenAiCompatible:
    """Shared transport for every chat-completions clone.

    Groq and OpenAI speak the same wire format, so the difference between them
    is an endpoint and a key name, both of which are config.
    """

    name = ""

    def __init__(self, cfg: LlmConfig) -> None:
        self.cfg = cfg
        self.endpoint = cfg.endpoints[self.name]
        self.model = cfg.model or self.endpoint.model

    def _headers(self) -> dict[str, str]:
        key = os.environ.get(self.endpoint.api_key_env, "")
        if not key:
            raise LomasError(
                f"{self.endpoint.api_key_env} is not set. Export it, or use "
                "llm.provider: offline."
            )
        return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    def _payload(self, messages: list[Message], stream: bool, **options) -> dict:
        return {
            "model": options.get("model") or self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": options.get("temperature", self.cfg.temperature),
            "max_tokens": options.get("max_tokens", self.cfg.max_tokens),
            "stream": stream,
        }

    def complete(self, messages: list[Message], **options) -> Completion:
        import httpx

        response = httpx.post(
            f"{self.endpoint.api_base}/chat/completions",
            headers=self._headers(),
            json=self._payload(messages, stream=False, **options),
            timeout=self.cfg.timeout_seconds,
        )
        response.raise_for_status()
        body = response.json()
        choice = body["choices"][FIRST]
        usage = body.get("usage", {})
        return Completion(
            text=choice["message"]["content"].strip(),
            provider=self.name,
            model=body.get("model", self.model),
            usage=Usage(
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
            ),
            finish_reason=choice.get("finish_reason", "stop"),
        )

    def stream(self, messages: list[Message], **options) -> Iterator[str]:
        import httpx

        with httpx.stream(
            "POST",
            f"{self.endpoint.api_base}/chat/completions",
            headers=self._headers(),
            json=self._payload(messages, stream=True, **options),
            timeout=self.cfg.timeout_seconds,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith(DATA_PREFIX):
                    continue
                chunk = line[len(DATA_PREFIX):]
                if chunk == DONE:
                    return
                delta = json.loads(chunk)["choices"][FIRST].get("delta", {})
                if delta.get("content"):
                    yield delta["content"]
