from repobot.ingestion.repo_map import build_repo_map
from repobot.models import Chunk


def chunk(path, symbol=None, parent=None):
    return Chunk("r", path, "python", 1, 2, "x", "function", symbol, parent)


def test_map_lists_files_with_top_level_symbols_only(tmp_path):
    (tmp_path / "README.md").write_text("# Demo\nA demo project.\n")
    chunks = [
        chunk("pkg/api.py", "get"),
        chunk("pkg/api.py", "Client"),
        chunk("pkg/api.py", "Client.send", parent="Client"),  # methods are omitted
        chunk("docs/guide.md"),
    ]
    repo_map = build_repo_map(tmp_path, chunks)

    assert "# Demo" in repo_map
    assert "- pkg/api.py: get, Client" in repo_map
    assert "Client.send" not in repo_map
    assert "- docs/guide.md" in repo_map


def test_map_is_truncated_to_budget(tmp_path):
    chunks = [chunk(f"pkg/module_{i:03d}.py", f"func_{i}") for i in range(500)]
    repo_map = build_repo_map(tmp_path, chunks, max_chars=1000)

    assert len(repo_map) <= 1100
    assert "more files" in repo_map
