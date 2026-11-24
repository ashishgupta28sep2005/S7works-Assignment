"""Opt-in live Groq smoke test. Disabled during ordinary pytest runs."""

from __future__ import annotations

import os

import pytest

from app.config.llm import create_groq_client
from app.config.settings import get_settings
from app.rag.embeddings import HuggingFaceEmbedder
from app.rag.retriever import TranscriptRetriever
from app.rag.vector_store import PersistentFaissStore
from app.tools.qualitative_analysis import QualitativeAnalysisTool


def _integration_enabled() -> bool:
    get_settings()  # Loads the ignored local .env without printing its values.
    return (os.getenv("RUN_GROQ_INTEGRATION_TESTS") or "false").strip().lower() == "true"


@pytest.mark.skipif(not _integration_enabled(), reason="Set RUN_GROQ_INTEGRATION_TESTS=true to make one live Groq request")
def test_configured_groq_model_accepts_structured_output():
    settings = get_settings()
    embedder = HuggingFaceEmbedder()
    store = PersistentFaissStore.open(
        settings.vectorstore_dir,
        dimension=embedder.dimension,
        embedding_model=settings.embedding_model,
    )
    client = create_groq_client(settings)
    tool = QualitativeAnalysisTool(TranscriptRetriever(store, embedder), llm=client)
    result = tool.analyze(
        "Summarize management's margin commentary.",
        reporting_periods=["Q4 FY2025-26"],
    )
    assert result.analysis_question == "Summarize management's margin commentary."
    assert result.sources or result.limitations
