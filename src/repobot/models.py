"""Core data types shared across ingestion, indexing, and retrieval."""

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

ChunkKind = Literal["function", "class", "method", "type", "section", "module", "block"]


@dataclass(frozen=True)
class RepoRef:
    owner: str
    name: str
    sha: str
    local_path: Path = field(compare=False)

    @property
    def repo_id(self) -> str:
        # Also the Qdrant collection name: one collection per repo@commit.
        return f"{self.owner}__{self.name}__{self.sha[:12]}"

    def blob_url(self, path: str, start_line: int, end_line: int) -> str:
        return (
            f"https://github.com/{self.owner}/{self.name}/blob/{self.sha}/{path}"
            f"#L{start_line}-L{end_line}"
        )


@dataclass
class Chunk:
    repo_id: str
    path: str  # POSIX path relative to repo root
    language: str
    start_line: int  # 1-based, inclusive
    end_line: int
    text: str
    kind: ChunkKind = "block"
    symbol: str | None = None  # e.g. "Session.request"
    parent: str | None = None  # enclosing class, if any

    @property
    def chunk_id(self) -> str:
        # Deterministic UUID so re-ingesting the same commit overwrites instead of duplicating.
        key = f"{self.repo_id}:{self.path}:{self.start_line}-{self.end_line}"
        return str(uuid.uuid5(uuid.NAMESPACE_URL, key))

    @property
    def label(self) -> str:
        return f"{self.path}:{self.start_line}-{self.end_line}"

    def embedding_text(self) -> str:
        """Text that gets embedded: a small header of metadata + the code itself.

        The header lets identifier/path queries match even when the body alone wouldn't.
        """
        header = f"# file: {self.path}"
        if self.symbol:
            header += f" | {self.kind}: {self.symbol}"
        return f"{header}\n{self.text}"


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float
