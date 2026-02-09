"""
LLM Client - Multi-provider LLM wrapper using LiteLLM.
Supports OpenAI, Anthropic, Google, Ollama, and any LiteLLM-compatible provider.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import litellm
from tenacity import retry, stop_after_attempt, wait_exponential

from src.core.config import settings
from src.core.logging import get_logger

logger = get_logger(__name__)

# Configure LiteLLM
litellm.drop_params = True  # Don't error on unsupported params
litellm.set_verbose = False


class LLMClient:
    """
    Async LLM client with retry logic, rate limiting, and multi-provider support.
    """

    def __init__(
        self,
        model: str = "gpt-4-turbo",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        timeout: int = 60,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._semaphore = asyncio.Semaphore(settings.max_parallel_conversations)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: str | None = None,
    ) -> str:
        """
        Generate a completion from the LLM.

        Args:
            prompt: The user message/prompt.
            system_prompt: Optional system message.
            temperature: Override default temperature.
            max_tokens: Override default max tokens.
            response_format: Set to "json" for JSON mode.

        Returns:
            The generated text response.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        return await self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: str | None = None,
    ) -> str:
        """
        Multi-turn chat completion.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            temperature: Override default temperature.
            max_tokens: Override default max tokens.
            response_format: Set to "json" for JSON mode.

        Returns:
            The generated text response.
        """
        async with self._semaphore:
            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature or self.temperature,
                "max_tokens": max_tokens or self.max_tokens,
                "timeout": self.timeout,
            }

            if response_format == "json":
                kwargs["response_format"] = {"type": "json_object"}

            logger.debug(
                "llm_request",
                model=self.model,
                num_messages=len(messages),
            )

            response = await litellm.acompletion(**kwargs)
            content = response.choices[0].message.content

            logger.debug(
                "llm_response",
                model=self.model,
                tokens_used=response.usage.total_tokens if response.usage else None,
            )

            return content

    async def generate_json(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> dict | list:
        """
        Generate and parse a JSON response.

        Returns:
            Parsed JSON as dict or list.
        """
        response = await self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            response_format="json",
        )

        # Clean response - strip markdown code fences if present
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        return json.loads(cleaned)


class LLMClientFactory:
    """Factory for creating LLM clients with common configurations."""

    _clients: dict[str, LLMClient] = {}

    @classmethod
    def get_client(
        cls,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> LLMClient:
        """Get or create a cached LLM client."""
        key = f"{model}_{temperature}_{max_tokens}"
        if key not in cls._clients:
            cls._clients[key] = LLMClient(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        return cls._clients[key]

    @classmethod
    def persona_generator(cls) -> LLMClient:
        return cls.get_client(
            model=settings.persona_generator_model,
            temperature=0.9,
            max_tokens=4000,
        )

    @classmethod
    def user_simulator(cls) -> LLMClient:
        return cls.get_client(
            model=settings.user_simulator_model,
            temperature=0.8,
            max_tokens=500,
        )

    @classmethod
    def quality_judge(cls) -> LLMClient:
        return cls.get_client(
            model=settings.quality_judge_model,
            temperature=0.1,
            max_tokens=2000,
        )
