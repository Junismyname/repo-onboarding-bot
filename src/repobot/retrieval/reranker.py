"""Cross-encoder re-ranking: slower but more accurate than the first-pass search."""

from repobot.config import settings
from repobot.models import ScoredChunk


class Reranker:
    def __init__(self, model_name: str = settings.rerank_model, device: str = settings.rerank_device):
        import torch
        from sentence_transformers import CrossEncoder

        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.model = CrossEncoder(
            model_name, device=device, max_length=512, model_kwargs={"torch_dtype": dtype}
        )

    def rerank(self, query: str, candidates: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        if not candidates:
            return []
        pairs = [(query, c.chunk.embedding_text()) for c in candidates]
        scores = self.model.predict(pairs, batch_size=16)
        ranked = sorted(zip(candidates, scores, strict=True), key=lambda x: x[1], reverse=True)
        return [ScoredChunk(c.chunk, float(score)) for c, score in ranked[:top_k]]
