"""Prompt construction.

Code is presented grouped by file and in line order (see retrieval.structure): with chunks in
relevance order, the model answered "the second class in model.py" by counting prompt order.
"""

import re

from repobot.intent import is_command_question
from repobot.models import ScoredChunk

SYSTEM_PROMPT = """\
You are a senior engineer helping a developer understand an unfamiliar codebase.

Rules:
- Answer ONLY from the repository context below. Do not invent files, functions, or behavior.
- Cite every claim with its source label in square brackets, e.g. [src/app/db.py:10-42].
- If the context does not contain the answer, say so plainly and suggest which files or
  symbols to look at next.
- Be concise. Include short code snippets only when they clarify the answer."""

# Only command questions (repobot.intent) get the how-to instruction below: added to every
# prompt, it pulled citations toward docs on "how does X work?" questions too
# (answers citing an expected file: 72% -> 61-67%).
HOWTO_INSTRUCTION = (
    # A user asked "how do I run this repo?" and got one cited sentence per step, no
    # commands, although the README's whole Getting Started section was in the prompt.
    "This question is about commands: answer with numbered steps and copy the exact commands "
    "from the docs above into ```bash blocks, like\n"
    # The example is a placeholder on purpose: a concrete example command gets copied into
    # answers for repos where it doesn't apply.
    "```bash\n<command exactly as written in the docs>\n```\n"
    "Don't replace a documented command with a description or with calls derived from the code.\n"
)


def format_chunk(scored: ScoredChunk) -> str:
    c = scored.chunk
    title = f"[{c.label}]" + (f" {c.kind} {c.symbol}" if c.symbol else "")
    # A README section has its own ``` blocks; a longer outer fence keeps them from closing it.
    longest_run = max((len(run) for run in re.findall(r"`{3,}", c.text)), default=2)
    fence = "`" * max(3, longest_run + 1)
    return f"{title}\n{fence}{c.language}\n{c.text}\n{fence}"


def build_messages(
    question: str,
    repo_map: str,
    chunks: list[ScoredChunk],
    outlines: list[str] = (),
    history: list[dict] = (),
    standalone: str | None = None,
) -> list[dict]:
    """System prompt, earlier turns (plain questions/answers), then this turn with context.

    Earlier turns are sent without their retrieved code, which keeps the prompt (and the KV
    cache it occupies) from growing with every turn.
    """
    context = "\n\n".join(format_chunk(c) for c in chunks)
    outline_section = ""
    if outlines:
        outline_section = (
            "## Outline of the files named in the question\n"
            "Authoritative order of definitions; use it for any question about order or "
            "position (first, second, last, ...).\n\n" + "\n\n".join(outlines) + "\n\n"
        )
    if standalone and standalone != question:
        question = f"{question}\n(In context, this means: {standalone})"
    # The outline goes right before the question: placed before ~6k tokens of code, the model
    # often ignored it (structure eval: 76% at the top of the prompt, 87% here).
    user = (
        f"## Repository overview\n{repo_map or '(not available)'}\n\n"
        f"## Retrieved code (grouped by file, in line order)\n{context or '(nothing retrieved)'}\n\n"
        f"{outline_section}"
        f"## Question\n{question}\n\n"
        # Repeated here on purpose: small models follow instructions near the end of a long
        # prompt far better than ones ~6k tokens back in the system message. The closing
        # "Sources:" line took answers with citations from 67% to 100% in the answer eval.
        "Answer from the retrieved code above. After each claim, cite its source label in "
        "square brackets exactly as shown, e.g. [src/pkg/module.py:10-42].\n"
        f"{HOWTO_INSTRUCTION if is_command_question(standalone or question) else ''}"
        "Finish with a final line `Sources:` followed by the labels of every snippet you used."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        *history,
        {"role": "user", "content": user},
    ]
