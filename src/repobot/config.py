"""Central settings. Override any field with an env var prefixed REPOBOT_ (or in .env)."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="REPOBOT_", extra="ignore")

    # LLM server (vLLM or Ollama; both speak the OpenAI API)
    llm_base_url: str = "http://localhost:8000/v1"
    llm_api_key: str = "not-needed"
    llm_model: str = "qwen2.5-coder-7b"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1024
    llm_max_model_len: int = 16384  # must match vLLM's --max-model-len (VLLM_MAX_LEN)
    llm_tokenizer: str = "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ"  # for prompt token budgeting

    # Embedding + re-ranking models
    embed_model: str = "Alibaba-NLP/gte-modernbert-base"  # 149M params, 768-d, trained on code too
    embed_device: str = "cuda"
    embed_batch_size: int = 64
    sparse_model: str = "Qdrant/bm25"
    rerank_model: str = "BAAI/bge-reranker-base"
    # Measured: 50 candidates take ~15 s on CPU vs ~160 ms on GPU (fp16, ~0.6 GB VRAM).
    rerank_device: str = "cuda"

    # Vector store
    qdrant_url: str = "http://localhost:6333"

    # Ingestion limits
    data_dir: Path = Path("data")
    max_repo_mb: int = 200
    max_file_kb: int = 512
    max_chunks: int = 60_000

    # Retrieval. Defaults chosen from the eval (see README "Retrieval ablations"):
    # dense search over code beat hybrid and re-ranked variants on recall@1 / MRR.
    retrieval_mode: str = "dense"  # "dense" | "hybrid"
    use_reranker: bool = False
    retrieve_k: int = 50  # candidates per search (re-ranker input, eval ranking depth)
    code_k: int = 10  # code chunks sent to the LLM
    docs_k: int = 3  # docs chunks sent to the LLM (searched separately so prose can't crowd out code)


settings = Settings()
