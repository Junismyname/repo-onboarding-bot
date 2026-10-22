"""Client for the LLM server (vLLM or Ollama via the OpenAI-compatible API)."""

import logging
from collections.abc import AsyncIterator

from openai import AsyncOpenAI

from repobot.config import settings

logger = logging.getLogger("uvicorn.error")
_client = AsyncOpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key)


async def stream_answer(messages: list[dict]) -> AsyncIterator[str]:
    stream = await _client.chat.completions.create(
        model=settings.llm_model,
        messages=messages,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        stream=True,
    )
    async for event in stream:
        if event.choices and (delta := event.choices[0].delta.content):
            yield delta


async def served_max_model_len() -> int | None:
    """The context length the server was started with (vLLM reports it; Ollama doesn't)."""
    try:
        models = await _client.models.list()
    except Exception:  # server not up yet: keep the configured value
        logger.exception("couldn't read max_model_len from %s", settings.llm_base_url)
        return None
    for model in models.data:
        if model.id == settings.llm_model:
            return getattr(model, "max_model_len", None)
    return None


async def complete(messages: list[dict], max_tokens: int) -> str:
    """Short, deterministic, non-streamed completion (used for helper calls)."""
    response = await _client.chat.completions.create(
        model=settings.llm_model,
        messages=messages,
        temperature=0.0,
        max_tokens=max_tokens,
    )
    return (response.choices[0].message.content or "").strip()
