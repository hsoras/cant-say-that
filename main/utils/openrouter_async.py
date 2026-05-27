# openrouter_async.py
"""
Async OpenRouter client using the OpenAI Python SDK.
Used by the MCTS engine for parallel attacker/target/judge API calls.
"""
import logging
import os

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


def _get_async_client() -> AsyncOpenAI:
    """Create an AsyncOpenAI client pointed at OpenRouter."""
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    base_url = os.environ.get(
        "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
    )
    if not api_key.strip():
        raise ValueError(
            "OPENROUTER_API_KEY is not set. Export it with your OpenRouter API key."
        )
    return AsyncOpenAI(api_key=api_key, base_url=base_url)


async def async_chat_completion(
    client: AsyncOpenAI,
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 0.7,
    max_tokens: int | None = None,
) -> str:
    """
    Single async chat completion call via OpenRouter.

    Returns the assistant's message content as a string.
    Raises on API errors (caller should handle retries if needed).
    """
    kwargs: dict = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    logger.debug(
        "OpenRouter async | model=%s | n_messages=%d | temp=%.1f",
        model,
        len(messages),
        temperature,
    )

    response = await client.chat.completions.create(**kwargs)
    content = response.choices[0].message.content
    if content is None:
        raise RuntimeError(
            f"OpenRouter async response missing content: {response!r}"
        )
    text = content.strip()
    logger.debug(
        "OpenRouter async done | model=%s | %d chars",
        model,
        len(text),
    )
    return text
