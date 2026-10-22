"""Ingestion pipeline: GitHub URL -> chunks -> embeddings -> Qdrant collection."""

from collections.abc import Callable
from typing import Literal

from repobot.config import settings
from repobot.errors import RepoTooLargeError
from repobot.ingestion.chunker import chunk_by_lines, chunk_file
from repobot.ingestion.clone import clone_repo
from repobot.ingestion.filters import detect_language, iter_source_files
from repobot.ingestion.repo_map import build_repo_map, repo_map_exists, save_repo_map
from repobot.models import Chunk, RepoRef
from repobot.services import Components

ProgressCallback = Callable[[str, float], None]
Chunking = Literal["syntax", "lines"]
# Chunks per encode() call. sentence-transformers sorts each call's inputs by length before
# batching, so large groups waste less padding: 512-chunk groups embed ~1.6x faster than
# 64-chunk groups (measured on axios, median chunk 387 tokens).
EMBED_GROUP = 512


def index_name(repo: RepoRef, chunking: Chunking = "syntax") -> str:
    """Qdrant collection for a repo. Line-window indexes exist only for eval comparisons."""
    return repo.repo_id if chunking == "syntax" else f"{repo.repo_id}--lines"


def is_indexed(store, repo_id: str) -> bool:
    """True once ingestion finished. The collection exists from the first upsert on, so on its
    own it would also match a half-built index (ingestion still running, or killed midway);
    the repo map is written only after the last chunk is stored."""
    return repo_map_exists(repo_id) and store.collection_exists(repo_id)


def ingest_repo(
    url: str,
    components: Components,
    on_progress: ProgressCallback = lambda stage, progress: None,
    chunking: Chunking = "syntax",
    sha: str | None = None,
) -> RepoRef:
    store, dense, sparse = components.store, components.dense, components.sparse

    on_progress("cloning", 0.0)
    repo = clone_repo(url, settings.data_dir, settings.max_repo_mb, sha)
    collection = index_name(repo, chunking)
    if chunking == "syntax":
        done = is_indexed(store, collection)
    else:  # eval-only line-window index: no repo map
        done = store.collection_exists(collection)
    if done:
        on_progress("done", 1.0)
        return repo  # this commit is already indexed
    if store.collection_exists(collection):
        store.delete_collection(collection)  # leftover from an interrupted ingestion

    on_progress("chunking", 0.1)
    chunks: list[Chunk] = []
    for path in iter_source_files(repo.local_path, settings.max_file_kb):
        rel_path = path.relative_to(repo.local_path).as_posix()
        source = path.read_text(encoding="utf-8", errors="replace")
        split = chunk_file if chunking == "syntax" else chunk_by_lines
        chunks.extend(split(source, rel_path, detect_language(path), repo.repo_id))
        if len(chunks) > settings.max_chunks:
            raise RepoTooLargeError(f"More than {settings.max_chunks} chunks; repo too large")

    on_progress("embedding", 0.2)
    store.create_collection(collection, dense.dim)
    try:
        for i in range(0, len(chunks), EMBED_GROUP):
            batch = chunks[i : i + EMBED_GROUP]
            texts = [c.embedding_text() for c in batch]
            store.upsert(
                collection, batch, dense.embed_documents(texts), sparse.embed_documents(texts)
            )
            on_progress("embedding", 0.2 + 0.8 * (i + len(batch)) / len(chunks))
    except Exception:
        store.delete_collection(collection)
        raise

    if chunking == "syntax":
        # Written last: its presence marks the index as complete (see is_indexed).
        save_repo_map(repo.repo_id, build_repo_map(repo.local_path, chunks))
    on_progress("done", 1.0)
    return repo
