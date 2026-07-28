"""Provider-neutral LLM implementations and registry."""

from __future__ import annotations

import json
from typing import Any

import httpx

from server.config import LLMProviderConfig
from server.interfaces import LLMProvider
from server.mock_drivers import MockLLMProvider


class LLMProviderError(Exception):
    """Raised when an LLM provider cannot complete a request."""


class OpenAICompatibleProvider:
    """Generic async provider for OpenAI-compatible chat completions APIs."""

    def __init__(
        self,
        name: str,
        config: LLMProviderConfig,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.name = name
        self.config = config
        self._http_client = http_client
        self._owns_http_client = http_client is None

    async def close(self) -> None:
        if self._owns_http_client and self._http_client is not None:
            await self._http_client.aclose()

    async def generate(self, prompt: str, system_prompt: str | None = None, **kwargs: Any) -> str:
        clean = " ".join(prompt.split()).strip()
        if not clean:
            raise LLMProviderError("Prompt cannot be empty.")

        messages: list[dict[str, str]] = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": " ".join(system_prompt.split()).strip()})
        messages.append({"role": "user", "content": clean})

        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "stream": False,
        }
        url = f"{self.config.base_url.rstrip('/')}/chat/completions"
        headers = {"Content-Type": "application/json"}
        api_key = self.config.get_api_key()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        try:
            response = await self._client().post(url, headers=headers, json=payload)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise LLMProviderError(f"{self.name} HTTP error {exc.response.status_code}: {exc.response.text}") from exc
        except httpx.HTTPError as exc:
            raise LLMProviderError(f"{self.name} request failed: {exc}") from exc

        try:
            data = response.json()
        except json.JSONDecodeError as exc:
            raise LLMProviderError(f"{self.name} returned non-JSON response.") from exc

        choices = data.get("choices") if isinstance(data, dict) else None
        if not isinstance(choices, list) or not choices:
            raise LLMProviderError(f"{self.name} response missing choices.")
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise LLMProviderError(f"{self.name} response content is empty.")
        return content.strip()

    def _client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=self.config.timeout_seconds)
        return self._http_client


class LLMProviderRegistry:
    """Explicit provider registry keyed by configured profile name."""

    def __init__(self) -> None:
        self._providers: dict[str, LLMProvider] = {}

    def register(self, name: str, provider: LLMProvider) -> None:
        self._providers[name] = provider

    def get(self, name: str) -> LLMProvider:
        try:
            return self._providers[name]
        except KeyError as exc:
            raise LLMProviderError(f"LLM provider not registered: {name}") from exc

    def names(self) -> list[str]:
        return sorted(self._providers)

    async def close_all(self) -> None:
        for provider in self._providers.values():
            close = getattr(provider, "close", None)
            if close is not None:
                await close()


def build_llm_registry(profiles: dict[str, LLMProviderConfig]) -> LLMProviderRegistry:
    registry = LLMProviderRegistry()
    for name, profile in profiles.items():
        if profile.provider == "mock":
            registry.register(name, MockLLMProvider())
        else:
            registry.register(name, OpenAICompatibleProvider(name=name, config=profile))
    return registry
