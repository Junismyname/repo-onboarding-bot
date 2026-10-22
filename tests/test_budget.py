from repobot.generation import budget
from repobot.generation.budget import fit_to_budget
from repobot.generation.prompts import format_chunk
from repobot.models import Chunk, ScoredChunk


def sc(name, score, size=100):
    return ScoredChunk(Chunk("r", f"{name}.py", "python", 1, 2, "x" * size), score)


def fake_count(messages):
    return sum(len(m["content"]) for m in messages)


def build(chunks):
    return [{"role": "user", "content": "".join(c.chunk.text for c in chunks)}]


def test_drops_lowest_scoring_chunks_until_it_fits(monkeypatch):
    monkeypatch.setattr(budget, "count_tokens", fake_count)
    chunks = [sc("pinned", 1.0), sc("good", 0.8), sc("weak", 0.3), sc("ok", 0.6)]
    messages, kept = fit_to_budget(build, chunks, budget=250)
    assert [c.chunk.path for c in kept] == ["pinned.py", "good.py"]
    assert fake_count(messages) <= 250


def test_keeps_everything_when_it_fits(monkeypatch):
    monkeypatch.setattr(budget, "count_tokens", fake_count)
    chunks = [sc("a", 0.5), sc("b", 0.4)]
    _, kept = fit_to_budget(build, chunks, budget=10_000)
    assert kept == chunks


def test_chunk_with_code_fences_gets_a_longer_outer_fence():
    readme = "## Setup\n```bash\npip install -r requirements.txt\n```"
    text = format_chunk(ScoredChunk(Chunk("r", "README.md", "markdown", 1, 4, readme), 1.0))
    assert text.splitlines()[1] == "````markdown"
    assert text.splitlines()[-1] == "````"
    plain = format_chunk(ScoredChunk(Chunk("r", "a.py", "python", 1, 1, "x = 1"), 1.0))
    assert plain.splitlines()[1] == "```python"
