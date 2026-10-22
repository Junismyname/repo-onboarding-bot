"""A compact overview of the repo that is always included in the LLM prompt."""

from collections import defaultdict
from pathlib import Path

from repobot.config import settings
from repobot.models import Chunk

README_NAMES = ("README.md", "README.rst", "README.txt", "README")
README_LINES = 40
MAX_SYMBOLS_PER_FILE = 12


def build_repo_map(repo_root: Path, chunks: list[Chunk], max_chars: int = 6000) -> str:
    """Markdown overview: README excerpt, then every indexed file with its top-level symbols.

    The file list is truncated to max_chars so the map never crowds out retrieved code.
    """
    parts = []
    readme = _readme_excerpt(repo_root)
    if readme:
        parts.append(f"### README (excerpt)\n{readme}")

    symbols: dict[str, list[str]] = defaultdict(list)
    for c in chunks:
        # Code symbols only: doc section titles (hundreds in a big README) would crowd them out.
        top_level_code = c.symbol and c.parent is None and c.kind != "section"
        if top_level_code and c.symbol not in symbols[c.path]:
            symbols[c.path].append(c.symbol)
        else:
            symbols.setdefault(c.path, [])

    lines = []
    for path in sorted(symbols):
        names = symbols[path]
        shown = ", ".join(names[:MAX_SYMBOLS_PER_FILE])
        if len(names) > MAX_SYMBOLS_PER_FILE:
            shown += f", … (+{len(names) - MAX_SYMBOLS_PER_FILE})"
        lines.append(f"- {path}" + (f": {shown}" if shown else ""))

    header = f"### Files ({len(lines)} indexed)\n"
    budget = max_chars - sum(len(p) for p in parts) - len(header)
    kept, used = [], 0
    for line in lines:
        if used + len(line) + 1 > budget:
            kept.append(f"- … {len(lines) - len(kept)} more files")
            break
        kept.append(line)
        used += len(line) + 1
    parts.append(header + "\n".join(kept))
    return "\n\n".join(parts)


def _readme_excerpt(repo_root: Path) -> str:
    for name in README_NAMES:
        path = repo_root / name
        if path.is_file():
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            return "\n".join(lines[:README_LINES]).strip()
    return ""


def _map_path(repo_id: str) -> Path:
    return settings.data_dir / "maps" / f"{repo_id}.md"


def save_repo_map(repo_id: str, repo_map: str) -> None:
    path = _map_path(repo_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(repo_map, encoding="utf-8")


def repo_map_exists(repo_id: str) -> bool:
    return _map_path(repo_id).exists()


def load_repo_map(repo_id: str) -> str:
    path = _map_path(repo_id)
    return path.read_text(encoding="utf-8") if path.exists() else ""
