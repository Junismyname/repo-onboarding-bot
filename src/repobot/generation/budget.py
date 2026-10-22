"""Keep prompts inside the model's context window.

Prompts are ~5k tokens (median over the evals, max ~10k), but chunk sizes vary and a request
can ask for up to 30 code chunks. vLLM rejects a prompt that leaves no room for the answer, so
the lowest-scoring chunks are dropped until prompt + max output tokens fit. Chunks of a file
named in the question are pinned at score 1.0 and go last.
"""

import logging
from collections.abc import Callable
from functools import lru_cache

from repobot.config import settings
from repobot.models import ScoredChunk

logger = logging.getLogger("uvicorn.error")
CHARS_PER_TOKEN = 3.0  # fallback estimate (code is token-dense) if the tokenizer can't load
SAFETY_MARGIN = 256


@lru_cache
def _tokenizer():
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(settings.llm_tokenizer)
    except Exception:  # offline / Ollama setup without HF access: fall back to an estimate
        logger.exception("tokenizer %s unavailable; estimating tokens", settings.llm_tokenizer)
        return None


def count_tokens(messages: list[dict]) -> int:
    tokenizer = _tokenizer()
    if tokenizer is None:
        return int(sum(len(m["content"]) for m in messages) / CHARS_PER_TOKEN)
    encoded = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True
    )
    return len(encoded["input_ids"])


def prompt_budget() -> int:
    return settings.llm_max_model_len - settings.llm_max_tokens - SAFETY_MARGIN


def fit_to_budget(
    build: Callable[[list[ScoredChunk]], list[dict]],
    chunks: list[ScoredChunk],
    budget: int | None = None,
) -> tuple[list[dict], list[ScoredChunk]]:
    """Messages built from as many chunks as fit, and the chunks actually used."""
    budget = budget or prompt_budget()
    kept = list(chunks)
    messages = build(kept)
    while kept and count_tokens(messages) > budget:
        kept.remove(min(kept, key=lambda sc: sc.score))
        messages = build(kept)
    if len(kept) < len(chunks):
        logger.info("prompt over budget: dropped %d of %d chunks", len(chunks) - len(kept), len(chunks))
    return messages, kept
