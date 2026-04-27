"""Groq LLM client module.

Features:
- Send prompt and return text response
- Streaming token support
- Structured error handling
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, AsyncIterator

import httpx


class GroqClientError(Exception):
    """Base exception for Groq client failures."""


class GroqRequestError(GroqClientError):
    """Raised for transport and HTTP errors."""


class GroqResponseError(GroqClientError):
    """Raised when response payload is invalid or empty."""


@dataclass(slots=True)
class GroqClientConfig:
    """Runtime configuration for Groq API calls."""

    api_key: str
    model: str = "llama-3.1-8b-instant"
    base_url: str = "https://api.groq.com/openai/v1"
    timeout_s: float = 60.0
    temperature: float = 0.2
    max_tokens: int = 512

    @classmethod
    def from_env(cls) -> "GroqClientConfig":
        """Build config from environment variables."""
        return cls(
            api_key=os.getenv("GROQ_API_KEY", "").strip(),
            model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant").strip(),
            base_url=os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1").strip(),
            timeout_s=float(os.getenv("GROQ_TIMEOUT_SECONDS", "60")),
            temperature=float(os.getenv("GROQ_TEMPERATURE", "0.2")),
            max_tokens=int(os.getenv("GROQ_MAX_TOKENS", "512")),
        )


class GroqClient:
    """Async Groq chat-completions client."""

    def __init__(self, config: GroqClientConfig | None = None, http_client: httpx.AsyncClient | None = None) -> None:
        self.config = config or GroqClientConfig.from_env()
        if not self.config.api_key:
            raise GroqClientError("Missing GROQ_API_KEY for Groq client.")

        self._http_client = http_client or httpx.AsyncClient(timeout=self.config.timeout_s)
        self._owns_http_client = http_client is None

    async def close(self) -> None:
        """Close owned HTTP client resources."""
        if self._owns_http_client:
            await self._http_client.aclose()

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Send prompt and return full model response text."""
        payload = self._build_payload(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )
        data = await self._post_json("/chat/completions", payload)
        return self._extract_text(data)

    async def stream_generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Stream response tokens/chunks as they arrive."""
        payload = self._build_payload(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        url = f"{self.config.base_url.rstrip('/')}/chat/completions"
        headers = self._headers()

        try:
            async with self._http_client.stream("POST", url, headers=headers, json=payload) as response:
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    body = await response.aread()
                    message = body.decode(errors="ignore")
                    raise GroqRequestError(f"Groq API HTTP error {response.status_code}: {message}") from exc

                async for line in response.aiter_lines():
                    if not line:
                        continue
                    if not line.startswith("data:"):
                        continue

                    data_part = line[len("data:") :].strip()
                    if data_part == "[DONE]":
                        break

                    try:
                        event = json.loads(data_part)
                    except json.JSONDecodeError as exc:
                        raise GroqResponseError(f"Invalid streaming JSON event: {data_part}") from exc

                    token = self._extract_stream_delta(event)
                    if token:
                        yield token
        except httpx.RequestError as exc:
            raise GroqRequestError(f"Groq streaming request failed: {exc}") from exc

    def _build_payload(
        self,
        prompt: str,
        system_prompt: str | None,
        temperature: float | None,
        max_tokens: int | None,
        stream: bool,
    ) -> dict[str, Any]:
        clean_prompt = " ".join(prompt.split()).strip()
        if not clean_prompt:
            raise GroqClientError("Prompt cannot be empty.")

        messages: list[dict[str, str]] = []
        if system_prompt:
            clean_system = " ".join(system_prompt.split()).strip()
            if clean_system:
                messages.append({"role": "system", "content": clean_system})
        messages.append({"role": "user", "content": clean_prompt})

        return {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature if temperature is None else temperature,
            "max_tokens": self.config.max_tokens if max_tokens is None else max_tokens,
            "stream": stream,
        }

    async def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.config.base_url.rstrip('/')}/{path.lstrip('/')}"
        try:
            response = await self._http_client.post(url, headers=self._headers(), json=payload)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            message = exc.response.text
            raise GroqRequestError(
                f"Groq API HTTP error {exc.response.status_code}: {message}"
            ) from exc
        except httpx.RequestError as exc:
            raise GroqRequestError(f"Groq request failed: {exc}") from exc

        try:
            data = response.json()
        except ValueError as exc:
            raise GroqResponseError("Groq returned non-JSON response.") from exc

        if not isinstance(data, dict):
            raise GroqResponseError("Groq response payload is not a JSON object.")
        return data

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _extract_text(payload: dict[str, Any]) -> str:
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise GroqResponseError("Groq response missing choices.")

        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise GroqResponseError("Groq response content is empty.")
        return content.strip()

    @staticmethod
    def _extract_stream_delta(event: dict[str, Any]) -> str:
        choices = event.get("choices")
        if not isinstance(choices, list) or not choices:
            return ""
        first = choices[0]
        if not isinstance(first, dict):
            return ""
        delta = first.get("delta")
        if not isinstance(delta, dict):
            return ""
        content = delta.get("content")
        if isinstance(content, str):
            return content
        return ""


def create_groq_client() -> GroqClient:
    """Create Groq client from environment configuration."""
    return GroqClient(GroqClientConfig.from_env())

