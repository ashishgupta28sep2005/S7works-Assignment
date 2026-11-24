"""Natural-language retrieval over transcript chunks stored in FAISS."""

from __future__ import annotations

from app.config.settings import get_settings
from app.rag.embeddings import HuggingFaceEmbedder
from app.rag.models import RetrievalResponse
from app.rag.vector_store import PersistentFaissStore


class TranscriptRetriever:
    def __init__(
        self,
        store: PersistentFaissStore,
        embedder: HuggingFaceEmbedder,
        *,
        top_k: int | None = None,
        similarity_threshold: float | None = None,
    ) -> None:
        settings = get_settings()
        self.store = store
        self.embedder = embedder
        self.top_k = top_k if top_k is not None else settings.rag_retrieval_top_k
        if self.top_k < 1:
            raise ValueError("top_k must be at least 1")
        self.similarity_threshold = (
            similarity_threshold if similarity_threshold is not None else settings.rag_similarity_threshold
        )

    def retrieve(
        self,
        query: str,
        *,
        reporting_period: str | None = None,
        document_name: str | None = None,
        top_k: int | None = None,
    ) -> RetrievalResponse:
        if not query.strip():
            raise ValueError("Retrieval query cannot be empty")
        if self.store.size == 0:
            return RetrievalResponse(query=query, rejected_reason="Transcript vector index is empty.")
        vector = self.embedder.encode_query(query)
        limit = self.top_k if top_k is None else top_k
        if limit < 1:
            raise ValueError("top_k must be at least 1")
        matches = self.store.search(
            vector,
            top_k=limit,
            similarity_threshold=self.similarity_threshold,
            reporting_period=reporting_period,
            document_name=document_name,
        )
        if not matches:
            reason = "No chunks match the requested document or reporting-period filter."
        elif not any(hit.relevant for hit in matches):
            reason = (
                f"Top similarity {matches[0].score:.3f} is below configured threshold "
                f"{self.similarity_threshold:.3f}."
            )
        else:
            reason = None
        return RetrievalResponse(query=query, matches=matches, rejected_reason=reason)
