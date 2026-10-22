"""Multi-turn support: rewrite follow-ups into standalone questions, and trim history.

Retrieval only sees one string. For "What about the first one?" that string matches nothing
useful, so before retrieval the follow-up is rewritten with the conversation into
"What is the first class in src/model.py?". The answering model still gets the recent turns,
so its reply stays consistent with what it said before.
"""

import logging
import re

from repobot.generation.llm import complete

logger = logging.getLogger("uvicorn.error")

MAX_TURNS = 6  # messages (3 question/answer pairs) kept for rewriting and answering
MAX_ANSWER_CHARS = 1200  # earlier answers are trimmed; their sources lines dropped
MAX_QUESTION_CHARS = 2000  # same cap as a new question (AskRequest)
REWRITE_MAX_TOKENS = 80

REWRITE_PROMPT = """\
Rewrite the user's latest question so it can be understood without the conversation.
- Resolve references like "it", "that class", "the first one", "and the third?" using the conversation.
- Keep file names, class names and function names exactly as written.
- If the question is already self-contained, return it unchanged.
Reply with the rewritten question only."""

SOURCES_RE = re.compile(r"\n+Sources:.*\Z", re.DOTALL)


def trim_history(history: list[dict]) -> list[dict]:
    """Last MAX_TURNS messages starting with a question, each capped in length, answers
    without their 'Sources:' lines. Uncapped, three long turns could crowd out the code."""
    recent = history[-MAX_TURNS:]
    while recent and recent[0]["role"] != "user":
        recent = recent[1:]  # an answer without its question would confuse the model
    trimmed = []
    for msg in recent:
        content = msg["content"]
        limit = MAX_QUESTION_CHARS
        if msg["role"] == "assistant":
            content = SOURCES_RE.sub("", content)
            limit = MAX_ANSWER_CHARS
        if len(content) > limit:
            content = content[:limit] + " …"
        trimmed.append({"role": msg["role"], "content": content})
    return trimmed


async def standalone_question(history: list[dict], question: str) -> str:
    """The question rewritten to stand alone; unchanged when there is no history."""
    if not history:
        return question
    transcript = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}" for m in history
    )
    try:
        rewritten = await complete(
            [
                {"role": "system", "content": REWRITE_PROMPT},
                {
                    "role": "user",
                    "content": f"Conversation:\n{transcript}\n\nLatest question: {question}",
                },
            ],
            max_tokens=REWRITE_MAX_TOKENS,
        )
    except Exception:  # a failed rewrite must not block answering
        logger.exception("question rewrite failed; using the question as asked")
        return question
    rewritten = rewritten.strip().strip('"').strip()
    # Guard against the model answering instead of rewriting.
    return rewritten if 3 <= len(rewritten) <= 500 else question
