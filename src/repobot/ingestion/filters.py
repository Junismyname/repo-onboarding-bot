"""Decide which files in a repo are worth indexing."""

import os
from collections.abc import Iterator
from pathlib import Path

SKIP_DIRS = {
    ".git", "node_modules", "vendor", "third_party", "dist", "build", "__pycache__",
    ".venv", "venv", ".tox", ".mypy_cache", ".pytest_cache", "site-packages",
}
SKIP_FILES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock", "uv.lock",
    "Cargo.lock", "Pipfile.lock", "composer.lock",
}
LANGUAGE_BY_EXT = {
    ".py": "python",
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript",
    ".ts": "typescript", ".tsx": "tsx",
    ".go": "go", ".rs": "rust", ".java": "java",
    ".md": "markdown", ".rst": "rst", ".txt": "text",
    ".toml": "toml", ".yaml": "yaml", ".yml": "yaml", ".cfg": "ini", ".ini": "ini",
}


# Files that define how to build / test / run a project. Matched by name: indexing every .json
# would pull in data files (axios's docs/data/sponsors.json, lock files).
LANGUAGE_BY_NAME = {
    "Dockerfile": "dockerfile",
    "Makefile": "make",
    "makefile": "make",
    "package.json": "json",
}


def detect_language(path: Path) -> str | None:
    return LANGUAGE_BY_NAME.get(path.name) or LANGUAGE_BY_EXT.get(path.suffix.lower())


def is_binary(path: Path, sniff_bytes: int = 8192) -> bool:
    with path.open("rb") as f:
        return b"\0" in f.read(sniff_bytes)


def is_minified(path: Path) -> bool:
    return ".min." in path.name


def iter_source_files(repo_root: Path, max_file_kb: int) -> Iterator[Path]:
    """Yield indexable files, pruning skipped directories before descending into them."""
    for dirpath, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for filename in filenames:
            path = Path(dirpath) / filename
            if filename in SKIP_FILES or is_minified(path):
                continue
            if detect_language(path) is None:
                continue
            if path.stat().st_size > max_file_kb * 1024 or is_binary(path):
                continue
            yield path
