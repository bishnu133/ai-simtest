"""
Conversation Simulator - Orchestrates multi-turn conversations between personas and the target bot.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Any

import httpx

from src.core.llm_client import LLMClient, LLMClientFactory
from src.core.logging import get_logger
from src.models import BotConfig, Conversation, Persona, Turn

logger = get_logger(__name__)


class TargetBotClient:
    """
    HTTP client that talks to the user's chatbot under test.
    Supports OpenAI-compatible, Anthropic-compatible, and custom formats.
    """

    def __init__(self, config: BotConfig):
        self.config = config
        self._client = httpx.AsyncClient(timeout=config.timeout_seconds)

    async def send_message(
        self,
        message: str,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> tuple[str, float]:
        """
        Send a message to the target bot and return (response_text, latency_ms).
        """
        start = time.perf_counter()

        try:
            if self.config.request_format == "openai":
                response_text = await self._send_openai_format(message, conversation_history)
            elif self.config.request_format == "anthropic":
                response_text = await self._send_anthropic_format(message, conversation_history)
            else:
                response_text = await self._send_custom_format(message, conversation_history)

            latency_ms = (time.perf_counter() - start) * 1000
            return response_text, latency_ms

        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000
            logger.error("bot_request_failed", error=str(e), latency_ms=latency_ms)
            raise

    async def _send_openai_format(
        self, message: str, history: list[dict[str, str]] | None
    ) -> str:
        messages = list(history or [])
        messages.append({"role": "user", "content": message})

        headers = {"Content-Type": "application/json", **self.config.headers}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        payload = {"messages": messages, "stream": False}

        resp = await self._client.post(
            self.config.api_endpoint,
            json=payload,
            headers=headers,
        )
        resp.raise_for_status()
        data = resp.json()

        # Navigate response path (e.g., "choices.0.message.content")
        return self._extract_response(data, self.config.response_path)

    async def _send_anthropic_format(
        self, message: str, history: list[dict[str, str]] | None
    ) -> str:
        messages = list(history or [])
        messages.append({"role": "user", "content": message})

        headers = {
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
            **self.config.headers,
        }
        if self.config.api_key:
            headers["x-api-key"] = self.config.api_key

        payload = {"messages": messages, "max_tokens": 1024}

        resp = await self._client.post(
            self.config.api_endpoint,
            json=payload,
            headers=headers,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("content", [{}])[0].get("text", "")

    async def _send_custom_format(
        self, message: str, history: list[dict[str, str]] | None
    ) -> str:
        headers = {"Content-Type": "application/json", **self.config.headers}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        payload = {"message": message, "history": history or []}

        resp = await self._client.post(
            self.config.api_endpoint,
            json=payload,
            headers=headers,
        )
        resp.raise_for_status()
        data = resp.json()
        return self._extract_response(data, self.config.response_path)

    def _extract_response(self, data: dict, path: str) -> str:
        """Navigate a dot-separated path through nested dicts/lists."""
        current: Any = data
        for key in path.split("."):
            if isinstance(current, list):
                current = current[int(key)]
            elif isinstance(current, dict):
                current = current[key]
            else:
                raise ValueError(f"Cannot navigate path '{path}' through {type(current)}")
        return str(current)

    async def close(self):
        await self._client.aclose()


class ConversationSimulator:
    """
    Orchestrates multi-turn conversations between a simulated user persona
    and the target bot under test.
    """

    def __init__(
        self,
        user_simulator_llm: LLMClient | None = None,
        max_parallel: int = 10,
    ):
        self.user_llm = user_simulator_llm or LLMClientFactory.user_simulator()
        self._semaphore = asyncio.Semaphore(max_parallel)

    async def run_conversations(
        self,
        personas: list[Persona],
        bot_config: BotConfig,
        max_turns: int = 15,
    ) -> list[Conversation]:
        """
        Run conversations for all personas in parallel (bounded).
        """
        bot_client = TargetBotClient(bot_config)

        try:
            tasks = [
                self._run_single(persona, bot_client, max_turns)
                for persona in personas
            ]
            conversations = await asyncio.gather(*tasks, return_exceptions=True)

            results = []
            for i, conv in enumerate(conversations):
                if isinstance(conv, Exception):
                    logger.error(
                        "conversation_failed",
                        persona=personas[i].name,
                        error=str(conv),
                    )
                    # Create a failed conversation record
                    results.append(Conversation(
                        persona_id=personas[i].id,
                        errors=[{"error": str(conv)}],
                    ))
                else:
                    results.append(conv)

            logger.info(
                "conversations_completed",
                total=len(results),
                successful=sum(1 for c in results if not c.errors),
            )
            return results

        finally:
            await bot_client.close()

    async def _run_single(
        self,
        persona: Persona,
        bot_client: TargetBotClient,
        max_turns: int,
    ) -> Conversation:
        """Run a single conversation within the semaphore."""
        async with self._semaphore:
            return await self.simulate_conversation(persona, bot_client, max_turns)

    async def simulate_conversation(
        self,
        persona: Persona,
        bot_client: TargetBotClient,
        max_turns: int = 15,
    ) -> Conversation:
        """
        Simulate a single multi-turn conversation.
        """
        conversation = Conversation(
            persona_id=persona.id,
            start_time=datetime.now(timezone.utc),
        )

        # Track message history for the bot API
        bot_history: list[dict[str, str]] = []
        target_turns = min(max_turns, persona.target_conversation_turns)

        logger.info("conversation_start", persona=persona.name, target_turns=target_turns)

        for turn_num in range(target_turns):
            try:
                # Step 1: Generate user message
                user_message = await self._generate_user_message(
                    persona=persona,
                    conversation=conversation,
                )

                conversation.turns.append(Turn(
                    speaker="user",
                    message=user_message,
                    timestamp=datetime.now(timezone.utc),
                    metadata={"turn_number": turn_num},
                ))

                # Step 2: Send to target bot
                bot_history.append({"role": "user", "content": user_message})

                bot_response, latency = await bot_client.send_message(
                    message=user_message,
                    conversation_history=bot_history[:-1],  # History before this message
                )

                conversation.turns.append(Turn(
                    speaker="bot",
                    message=bot_response,
                    timestamp=datetime.now(timezone.utc),
                    latency_ms=latency,
                    metadata={"turn_number": turn_num},
                ))

                bot_history.append({"role": "assistant", "content": bot_response})

                # Step 3: Check if conversation should end naturally
                if self._should_end(conversation, persona):
                    logger.debug("conversation_ended_naturally", turn=turn_num)
                    break

            except Exception as e:
                conversation.errors.append({
                    "turn": turn_num,
                    "error": str(e),
                    "type": type(e).__name__,
                })
                logger.warning(
                    "conversation_turn_error",
                    persona=persona.name,
                    turn=turn_num,
                    error=str(e),
                )
                break

        conversation.end_time = datetime.now(timezone.utc)

        logger.info(
            "conversation_end",
            persona=persona.name,
            turns=conversation.turn_count,
            errors=len(conversation.errors),
        )

        return conversation

    async def _generate_user_message(
        self,
        persona: Persona,
        conversation: Conversation,
    ) -> str:
        """Generate the next user message based on the persona and conversation history."""

        history_text = conversation.format_history(max_turns=10) if conversation.turns else "This is the start of the conversation."

        prompt = f"""Conversation so far:
{history_text}

Generate the NEXT message this user would send. Output ONLY the message text, nothing else."""

        response = await self.user_llm.generate(
            prompt=prompt,
            system_prompt=persona.system_prompt,
        )

        return response.strip().strip('"')

    def _should_end(self, conversation: Conversation, persona: Persona) -> bool:
        """Determine if the conversation should end naturally."""
        if conversation.turn_count < 4:  # Minimum 2 exchanges
            return False

        # Check for natural endings
        if conversation.turns:
            last_user = next(
                (t.message.lower() for t in reversed(conversation.turns) if t.speaker == "user"),
                "",
            )
            endings = ["thank you", "thanks", "got it", "perfect", "that's all", "goodbye", "bye"]
            if any(phrase in last_user for phrase in endings):
                return True

        # Check for repeated bot refusals
        bot_messages = [t.message.lower() for t in conversation.turns if t.speaker == "bot"]
        refusal_phrases = ["i cannot", "i can't", "i'm unable", "i am unable", "i'm sorry, but i"]
        refusals = sum(
            1 for msg in bot_messages[-3:] if any(p in msg for p in refusal_phrases)
        )
        if refusals >= 2:
            return True

        return False