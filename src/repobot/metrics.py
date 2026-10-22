"""Retrieval metrics at file level: did the right files reach the LLM?"""

from collections.abc import Sequence


def unique_paths(paths: Sequence[str]) -> list[str]:
    """Dedupe while keeping rank order (many chunks can come from one file)."""
    return list(dict.fromkeys(paths))


def recall_at_k(retrieved: Sequence[str], expected: Sequence[str], k: int) -> float:
    """Fraction of expected files that appear in the top-k unique retrieved files."""
    if not expected:
        return 0.0
    top = set(unique_paths(retrieved)[:k])
    return sum(path in top for path in expected) / len(expected)


def reciprocal_rank(retrieved: Sequence[str], expected: Sequence[str]) -> float:
    """1 / rank of the first expected file (0 if none found)."""
    wanted = set(expected)
    for rank, path in enumerate(unique_paths(retrieved), start=1):
        if path in wanted:
            return 1.0 / rank
    return 0.0
