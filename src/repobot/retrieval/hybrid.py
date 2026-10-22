"""Query-time retrieval: separate searches over code and docs, merged into one context.

Prose docs match natural-language questions well, so in a single mixed search they push the
implementing code out of the top results. Searching each scope separately and giving each a
fixed budget keeps the code in the context while still including relevant docs.

When the question names a file ("the second class in model.py"), that file's outline is added
and its chunks are searched directly, and all code is presented in file order.
"""

from dataclasses import dataclass, field

from repobot.config import settings
from repobot.indexing.store import DENSE
from repobot.intent import is_command_question
from repobot.models import ScoredChunk
from repobot.retrieval.structure import in_file_order, mentioned_files, outline
from repobot.services import Components

SMALL_FILE_CHUNKS = 8  # files up to this many chunks are included whole when named
FILE_SEARCH_K = 6  # otherwise: best chunks within the named file
# Where projects define their commands. For "how do I run the tests?", semantic search fills
# the context with tests/*.py (code *about* testing) and misses the Makefile's `test:` target.
BUILD_FILES = {"Makefile", "makefile", "package.json", "Dockerfile"}
BUILD_FILE_K = 2  # best chunks per root-level build file, for command questions only
BUILD_FILE_SCORE = 0.95  # under the token budget: dropped after named files, before search hits


@dataclass
class Retrieved:
    chunks: list[ScoredChunk]
    outlines: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)  # files the question named


def retrieve(
    repo_id: str,
    question: str,
    components: Components,
    code_k: int = settings.code_k,
    docs_k: int = settings.docs_k,
) -> Retrieved:
    store = components.store
    query_dense = components.dense.embed_query(question)
    query_sparse = (
        components.sparse.embed_query(question) if settings.retrieval_mode == "hybrid" else None
    )

    def search(scope: str, k: int) -> list[ScoredChunk]:
        if k == 0:
            return []
        limit = settings.retrieve_k if components.reranker else k
        if query_sparse is not None:
            hits = store.hybrid_search(repo_id, query_dense, query_sparse, limit, scope)
        else:
            hits = store.search(repo_id, query_dense, DENSE, limit, scope)
        if components.reranker:
            hits = components.reranker.rerank(question, hits, k)
        return hits[:k]

    paths = store.paths(repo_id)
    files = mentioned_files(question, paths)
    pinned: list[ScoredChunk] = []
    outlines = []
    for path in files:
        chunks = store.file_chunks(repo_id, path)
        outlines.append(outline(path, chunks))
        if len(chunks) <= SMALL_FILE_CHUNKS:
            pinned += [ScoredChunk(c, 1.0) for c in chunks]
        else:
            pinned += store.search(repo_id, query_dense, DENSE, FILE_SEARCH_K, "all", path)

    if is_command_question(question):
        for path in paths:
            if path in BUILD_FILES and path not in files:  # root level only: no test fixtures
                hits = store.search(repo_id, query_dense, DENSE, BUILD_FILE_K, "all", path)
                pinned += [ScoredChunk(h.chunk, BUILD_FILE_SCORE) for h in hits]

    seen = {sc.chunk.chunk_id for sc in pinned}
    code = pinned + [sc for sc in search("code", code_k) if sc.chunk.chunk_id not in seen]
    docs = [sc for sc in search("docs", docs_k) if sc.chunk.chunk_id not in seen]
    return Retrieved(chunks=in_file_order(code, files) + docs, outlines=outlines, files=files)
