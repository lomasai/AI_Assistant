from __future__ import annotations

import os
from typing import Iterator

from lomas_core.errors import LomasError
from lomas_core.schema import LlmConfig
from lomas_llm.provider import PROVIDERS, split_system
from lomas_llm.types import Completion, Message, Usage

REFUSAL = "refusal"
TEXT_BLOCK = "text"
FALLBACK_BETA = "server-side-fallback-2026-07-01"
DEFAULT_FALLBACKS = "default"


@PROVIDERS.register("anthropic")
class AnthropicProvider:
    """Claude through the official SDK.

    Three things differ from the OpenAI-compatible providers and are easy to
    get wrong:

    - `temperature` is rejected on Opus 5, so it is never sent. Depth is
      controlled with `effort` instead, which is in config.
    - The system prompt is its own argument, not the first message.
    - A safety decline arrives as HTTP 200 with stop_reason "refusal", not as
      an exception, so `stop_reason` is checked before the text is read. With
      server-side fallbacks on, the API retries on another model inside the
      same call before it gets here.
    """

    name = "anthropic"

    def __init__(self, cfg: LlmConfig) -> None:
        self.cfg = cfg
        self.endpoint = cfg.endpoints[self.name]
        self.model = cfg.model or self.endpoint.model
        self._client = None

    def _ensure(self):
        if self._client is not None:
            return self._client
        try:
            import anthropic
        except ImportError as exc:
            raise LomasError(
                "the anthropic SDK is not installed. pip install anthropic, or "
                "use llm.provider: offline."
            ) from exc

        key = os.environ.get(self.endpoint.api_key_env, "")
        if not key:
            raise LomasError(f"{self.endpoint.api_key_env} is not set.")
        self._client = anthropic.Anthropic(api_key=key, timeout=self.cfg.timeout_seconds)
        return self._client

    def _request(self, messages: list[Message], **options) -> dict:
        system, rest = split_system(messages)
        request: dict = {
            "model": options.get("model") or self.model,
            "max_tokens": options.get("max_tokens", self.cfg.max_tokens),
            "messages": [{"role": m.role, "content": m.content} for m in rest],
            "output_config": {"effort": options.get("effort", self.endpoint.effort)},
        }
        if system:
            request["system"] = system
        if self.endpoint.server_side_fallback:
            request["betas"] = [FALLBACK_BETA]
            request["fallbacks"] = DEFAULT_FALLBACKS
        return request

    def _messages_api(self, client):
        return client.beta.messages if self.endpoint.server_side_fallback else client.messages

    def complete(self, messages: list[Message], **options) -> Completion:
        client = self._ensure()
        response = self._messages_api(client).create(**self._request(messages, **options))

        if response.stop_reason == REFUSAL:
            return Completion(text="", provider=self.name, model=response.model,
                              finish_reason=REFUSAL)

        text = "".join(block.text for block in response.content if block.type == TEXT_BLOCK)
        return Completion(
            text=text.strip(),
            provider=self.name,
            model=response.model,
            usage=Usage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            ),
            finish_reason=response.stop_reason or "end_turn",
        )

    def stream(self, messages: list[Message], **options) -> Iterator[str]:
        client = self._ensure()
        with self._messages_api(client).stream(**self._request(messages, **options)) as flow:
            yield from flow.text_stream
