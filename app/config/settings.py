"""Small environment-backed settings object used by local RAG components."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    llm_provider: str = "groq"
    llm_model: str = "llama-3.3-70b-versatile"
    groq_api_key: str | None = field(default=None, repr=False)
    groq_timeout_seconds: int = 45
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"
    embedding_batch_size: int = 32
    rag_chunk_size: int = 650
    rag_chunk_overlap: int = 100
    rag_similarity_threshold: float = 0.25
    rag_retrieval_top_k: int = 5
    cors_allowed_origins: tuple[str, ...] = ("http://localhost:5173",)
    vectorstore_dir: Path = PROJECT_ROOT / "vectorstore"
    model_cache_dir: Path = PROJECT_ROOT / ".cache" / "sentence-transformers"

    def __post_init__(self) -> None:
        if self.embedding_batch_size < 1:
            raise ValueError("EMBEDDING_BATCH_SIZE must be at least 1")
        if self.rag_chunk_size < 1:
            raise ValueError("RAG_CHUNK_SIZE must be at least 1")
        if not 0 <= self.rag_chunk_overlap < self.rag_chunk_size:
            raise ValueError("RAG_CHUNK_OVERLAP must be non-negative and smaller than RAG_CHUNK_SIZE")
        if not -1.0 <= self.rag_similarity_threshold <= 1.0:
            raise ValueError("RAG_SIMILARITY_THRESHOLD must be between -1 and 1")
        if self.rag_retrieval_top_k < 1:
            raise ValueError("RAG_RETRIEVAL_TOP_K must be at least 1")
        if self.groq_timeout_seconds < 1:
            raise ValueError("GROQ_TIMEOUT_SECONDS must be at least 1")
        for origin in self.cors_allowed_origins:
            parsed = urlsplit(origin)
            if (
                origin == "*"
                or parsed.scheme not in {"http", "https"}
                or not parsed.netloc
                or parsed.path
                or parsed.query
                or parsed.fragment
                or parsed.username
                or parsed.password
            ):
                raise ValueError("CORS_ALLOWED_ORIGINS must contain explicit HTTP(S) origins, not wildcards or URLs with paths.")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Read .env once; environment variables override the example defaults."""
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    return Settings(
        llm_provider=os.getenv("LLM_PROVIDER", Settings.llm_provider).strip().lower(),
        # GROQ_MODEL is the current provider-specific setting; retain LLM_MODEL as
        # a compatibility fallback for existing local configurations.
        llm_model=os.getenv("GROQ_MODEL", os.getenv("LLM_MODEL", Settings.llm_model)),
        groq_api_key=os.getenv("GROQ_API_KEY") or None,
        groq_timeout_seconds=int(os.getenv("GROQ_TIMEOUT_SECONDS", "45")),
        embedding_model=os.getenv("EMBEDDING_MODEL", Settings.embedding_model),
        embedding_device=os.getenv("EMBEDDING_DEVICE", Settings.embedding_device),
        embedding_batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", Settings.embedding_batch_size)),
        rag_chunk_size=int(os.getenv("RAG_CHUNK_SIZE", Settings.rag_chunk_size)),
        rag_chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", Settings.rag_chunk_overlap)),
        rag_similarity_threshold=float(os.getenv("RAG_SIMILARITY_THRESHOLD", Settings.rag_similarity_threshold)),
        rag_retrieval_top_k=int(os.getenv("RAG_RETRIEVAL_TOP_K", Settings.rag_retrieval_top_k)),
        cors_allowed_origins=tuple(
            origin.strip().rstrip("/")
            for origin in os.getenv(
                "CORS_ALLOWED_ORIGINS", ",".join(Settings.cors_allowed_origins),
            ).split(",")
            if origin.strip()
        ),
        vectorstore_dir=Path(os.getenv("VECTORSTORE_DIR", str(Settings.vectorstore_dir))).expanduser(),
        model_cache_dir=Path(os.getenv("MODEL_CACHE_DIR", str(Settings.model_cache_dir))).expanduser(),
    )
