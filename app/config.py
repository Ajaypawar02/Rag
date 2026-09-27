"""Application settings loaded from environment variables / `.env`.

All tunables live here so the rest of the codebase never reads `os.environ`
directly. Import the cached singleton with `from app.config import get_settings`.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Default models per provider, used when LLM_MODEL / LLM_FAST_MODEL are left empty.
# "main" answers the user; "fast" powers the lightweight routing/classification agents.
DEFAULT_MODELS: dict[str, dict[str, str]] = {
    "anthropic": {"main": "claude-sonnet-5", "fast": "claude-haiku-4-5-20251001"},
    "openai": {"main": "gpt-4o", "fast": "gpt-4o-mini"},
    "groq": {"main": "openai/gpt-oss-120b", "fast": "openai/gpt-oss-20b"},
}

DEFAULT_EMBEDDING_MODELS: dict[str, str] = {
    "huggingface": "sentence-transformers/all-MiniLM-L6-v2",
    "openai": "text-embedding-3-small",
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM -----------------------------------------------------------------
    llm_provider: Literal["anthropic", "openai", "groq"] = "anthropic"
    llm_model: str = ""
    llm_fast_model: str = ""
    llm_temperature: float = 0.0
    llm_max_tokens: int = 2048  # reasoning models (e.g. gpt-oss) spend part of this on hidden reasoning

    anthropic_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    groq_api_key: SecretStr | None = None

    # --- Embeddings ----------------------------------------------------------
    embedding_provider: Literal["huggingface", "openai"] = "huggingface"
    embedding_model: str = ""

    # --- Vector store (Chroma) -----------------------------------------------
    # "http": connect to the Chroma server (docker-compose). "local": embedded, on-disk.
    chroma_mode: Literal["http", "local"] = "http"
    chroma_persist_dir: Path = PROJECT_ROOT / ".chroma"
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    chroma_collection: str = "tech_docs"

    # --- Persistent memory (PostgreSQL) --------------------------------------
    postgres_url: str = "postgresql://rag:rag@localhost:5432/rag"

    # --- Ingestion -----------------------------------------------------------
    sample_docs_dir: Path = PROJECT_ROOT / "data" / "sample_docs"
    upload_dir: Path = PROJECT_ROOT / "data" / "uploads"
    chunk_size: int = 1000
    chunk_overlap: int = 150

    # --- Retrieval -----------------------------------------------------------
    retrieval_top_k: int = 5  # chunks passed to the LLM
    retrieval_fetch_k: int = 20  # candidates fetched per search method before fusion/reranking
    # Defaults chosen with scripts/eval_retrieval.py (see README benchmarks): semantic-leaning
    # fusion with a small RRF constant scored best; the cross-encoder reranker did not
    # improve accuracy on this corpus and adds ~100 ms, so it is opt-in.
    hybrid_alpha: float = 0.7  # 1.0 = pure semantic, 0.0 = pure keyword (BM25)
    rrf_k: int = 10  # reciprocal rank fusion damping constant; lower = top ranks count more
    rerank_enabled: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    filter_fallback: bool = True  # retry without metadata filters when they match nothing

    # --- Conversation memory -------------------------------------------------
    memory_strategy: Literal["sliding_window", "summary_buffer"] = "summary_buffer"
    window_size: int = 6  # messages kept verbatim
    summary_threshold: int = 10  # summarize once the thread exceeds this many messages
    compression_ratio: float = 0.3  # target summary length relative to summarized text

    # --- Services ------------------------------------------------------------
    api_url: str = "http://localhost:8080"
    max_upload_mb: int = 20

    # --- Observability (LangSmith) -------------------------------------------
    langsmith_tracing: bool = False
    langsmith_api_key: SecretStr | None = None
    langsmith_project: str = "conversational-rag"
    langsmith_endpoint: str | None = None  # e.g. https://eu.api.smith.langchain.com for the EU region

    @property
    def resolved_llm_model(self) -> str:
        return self.llm_model or DEFAULT_MODELS[self.llm_provider]["main"]

    @property
    def resolved_llm_fast_model(self) -> str:
        return self.llm_fast_model or DEFAULT_MODELS[self.llm_provider]["fast"]

    @property
    def resolved_embedding_model(self) -> str:
        return self.embedding_model or DEFAULT_EMBEDDING_MODELS[self.embedding_provider]


@lru_cache
def get_settings() -> Settings:
    return Settings()
