"""Heavy, long-lived objects (models, DB client), built once and shared by API and eval."""

import threading
from dataclasses import dataclass

from repobot.config import settings
from repobot.indexing.embedder import DenseEmbedder, SparseEmbedder
from repobot.indexing.store import VectorStore
from repobot.retrieval.reranker import Reranker


@dataclass
class Components:
    store: VectorStore
    dense: DenseEmbedder
    sparse: SparseEmbedder
    reranker: Reranker | None


_components: dict[bool, Components] = {}
# Two first requests arriving together would otherwise each load the models (2x VRAM).
_lock = threading.Lock()


def get_components(use_reranker: bool = settings.use_reranker) -> Components:
    if use_reranker not in _components:
        with _lock:
            if use_reranker not in _components:
                _components[use_reranker] = Components(
                    store=VectorStore(),
                    dense=DenseEmbedder(),
                    sparse=SparseEmbedder(),
                    reranker=Reranker() if use_reranker else None,
                )
    return _components[use_reranker]
