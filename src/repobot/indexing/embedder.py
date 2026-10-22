"""Dense (semantic) and sparse (BM25 keyword) embedders."""

from qdrant_client import models

from repobot.config import settings


class DenseEmbedder:
    def __init__(self, model_name: str = settings.embed_model, device: str = settings.embed_device):
        import torch
        from sentence_transformers import SentenceTransformer

        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.model = SentenceTransformer(
            model_name, device=device, model_kwargs={"torch_dtype": dtype}
        )
        # Cap input length: chunks are small, and long inputs spike VRAM next to vLLM.
        self.model.max_seq_length = 1024
        self.dim: int = self.model.get_embedding_dimension()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(
            texts, batch_size=settings.embed_batch_size, normalize_embeddings=True
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        # Some embedding models expect a query prefix (e.g. "query: "); check the model card.
        return self.model.encode(text, normalize_embeddings=True).tolist()


class SparseEmbedder:
    """BM25 term weights. Catches exact identifiers (`get_db_session`) that dense vectors blur."""

    def __init__(self, model_name: str = settings.sparse_model):
        from fastembed import SparseTextEmbedding

        self.model = SparseTextEmbedding(model_name)

    def embed_documents(self, texts: list[str]) -> list[models.SparseVector]:
        return [_to_qdrant(e) for e in self.model.embed(texts)]

    def embed_query(self, text: str) -> models.SparseVector:
        return _to_qdrant(next(iter(self.model.query_embed(text))))


def _to_qdrant(embedding) -> models.SparseVector:
    return models.SparseVector(
        indices=embedding.indices.tolist(), values=embedding.values.tolist()
    )
