"""Qdrant wrapper: one collection per repo@commit, with a dense and a sparse (BM25) vector."""

from dataclasses import asdict

from qdrant_client import QdrantClient, models

from repobot.config import settings
from repobot.models import Chunk, ScoredChunk

DENSE = "dense"
SPARSE = "bm25"
DOC_LANGUAGES = ["markdown", "rst", "text"]


def scope_filter(scope: str, path: str | None = None) -> models.Filter | None:
    """'code' = source files only, 'docs' = prose only, 'all' = no filter; optionally one file."""
    is_doc = models.FieldCondition(key="language", match=models.MatchAny(any=DOC_LANGUAGES))
    must = [models.FieldCondition(key="path", match=models.MatchValue(value=path))] if path else []
    must_not = []
    if scope == "code":
        must_not.append(is_doc)
    elif scope == "docs":
        must.append(is_doc)
    if not must and not must_not:
        return None
    return models.Filter(must=must or None, must_not=must_not or None)


class VectorStore:
    def __init__(self, url: str = settings.qdrant_url):
        self.client = QdrantClient(url=url, timeout=60)
        self._paths: dict[str, list[str]] = {}  # repo_id -> indexed file paths (immutable per commit)

    def collection_exists(self, repo_id: str) -> bool:
        return self.client.collection_exists(repo_id)

    def list_repos(self) -> list[str]:
        """Indexed repo ids (eval-only line-window indexes, named "...--lines", excluded)."""
        names = (c.name for c in self.client.get_collections().collections)
        return sorted(n for n in names if "--" not in n)

    def delete_collection(self, repo_id: str) -> None:
        self.client.delete_collection(repo_id)
        self._paths.pop(repo_id, None)

    def create_collection(self, repo_id: str, dense_dim: int) -> None:
        self.client.create_collection(
            repo_id,
            vectors_config={
                DENSE: models.VectorParams(size=dense_dim, distance=models.Distance.COSINE)
            },
            # IDF modifier: Qdrant computes the corpus-level IDF part of BM25 at query time.
            sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
        for field in ("path", "language"):
            self.client.create_payload_index(
                repo_id, field_name=field, field_schema=models.PayloadSchemaType.KEYWORD
            )

    def upsert(
        self,
        repo_id: str,
        chunks: list[Chunk],
        dense: list[list[float]],
        sparse: list[models.SparseVector],
    ) -> None:
        points = [
            models.PointStruct(id=c.chunk_id, vector={DENSE: d, SPARSE: s}, payload=asdict(c))
            for c, d, s in zip(chunks, dense, sparse, strict=True)
        ]
        self.client.upsert(repo_id, points=points, wait=True)

    def hybrid_search(
        self,
        repo_id: str,
        query_dense: list[float],
        query_sparse: models.SparseVector,
        limit: int,
        scope: str = "all",
    ) -> list[ScoredChunk]:
        """Dense and BM25 searches run inside Qdrant, merged with reciprocal rank fusion."""
        query_filter = scope_filter(scope)
        response = self.client.query_points(
            repo_id,
            prefetch=[
                models.Prefetch(query=query_dense, using=DENSE, limit=limit, filter=query_filter),
                models.Prefetch(query=query_sparse, using=SPARSE, limit=limit, filter=query_filter),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )
        return [ScoredChunk(Chunk(**p.payload), p.score) for p in response.points]

    def search(
        self,
        repo_id: str,
        query: list[float] | models.SparseVector,
        using: str,
        limit: int,
        scope: str = "all",
        path: str | None = None,
    ) -> list[ScoredChunk]:
        """Single-vector search (dense-only or BM25-only), optionally within one file."""
        response = self.client.query_points(
            repo_id,
            query=query,
            using=using,
            limit=limit,
            query_filter=scope_filter(scope, path),
            with_payload=True,
        )
        return [ScoredChunk(Chunk(**p.payload), p.score) for p in response.points]

    def paths(self, repo_id: str) -> list[str]:
        """Every indexed file path. Cached: a collection never changes once built."""
        if repo_id not in self._paths:
            payloads = self._scroll(repo_id, scroll_filter=None, fields=["path"])
            self._paths[repo_id] = sorted({p["path"] for p in payloads})
        return self._paths[repo_id]

    def file_chunks(self, repo_id: str, path: str) -> list[Chunk]:
        """All chunks of one file, in file order."""
        payloads = self._scroll(repo_id, scroll_filter=scope_filter("all", path), fields=True)
        return sorted((Chunk(**p) for p in payloads), key=lambda c: (c.start_line, c.end_line))

    def _scroll(self, repo_id: str, scroll_filter, fields) -> list[dict]:
        payloads, offset = [], None
        while True:
            points, offset = self.client.scroll(
                repo_id,
                scroll_filter=scroll_filter,
                limit=1000,
                offset=offset,
                with_payload=fields,
                with_vectors=False,
            )
            payloads.extend(p.payload for p in points)
            if offset is None:
                return payloads
