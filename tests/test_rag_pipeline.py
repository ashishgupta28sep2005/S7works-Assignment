"""Synthetic technical tests for transcript RAG; contains no TCS financial claims."""

from __future__ import annotations

import json

import numpy as np
import pytest

from app.ingestion.models import (
    DocumentMetadata, DocumentType, ExtractedPage, ProcessedDocument, ProcessingStatus,
)
from app.rag.embeddings import EmbeddingError, HuggingFaceEmbedder
from app.rag.models import TranscriptChunk
from app.rag.retriever import TranscriptRetriever
from app.rag.text_chunker import chunk_transcript
from app.rag.vector_store import (
    EmbeddingModelMismatch, PersistentFaissStore, VectorStoreError,
)


class FakeModel:
    def get_sentence_embedding_dimension(self):
        return 3

    def encode(self, texts, **kwargs):
        return np.array([[len(t), t.count("a") + 1, 1] for t in texts], dtype=np.float32)

    class tokenizer:
        @staticmethod
        def encode(text, add_special_tokens=False):
            return text.split()


def make_doc(*, text: str, doc_type=DocumentType.EARNINGS_TRANSCRIPT, pages=None):
    metadata = DocumentMetadata(
        original_filename="synthetic.pdf", source_path="synthetic.pdf", sha256="a" * 64,
        document_type=doc_type, company="Example Corp", reporting_period="Q1 FY2027",
        management_speakers=["Chief Executive Officer"],
    )
    return ProcessedDocument(
        metadata=metadata,
        pages=pages or [ExtractedPage(page_number=1, text=text)],
        processing_status=ProcessingStatus.PROCESSED,
    )


def make_chunk(chunk_id="chunk-1", page=1, name="synthetic.pdf", digest="a" * 64):
    return TranscriptChunk(
        company="Example Corp", document_name=name, page_number=page,
        chunk_id=chunk_id, original_text="Management discussed operating margins.",
        document_hash=digest,
    )


def test_chunker_keeps_page_and_speaker_provenance_and_skips_cover():
    doc = make_doc(text="", pages=[
        ExtractedPage(page_number=1, text="Example Corp earnings call"),
        ExtractedPage(page_number=2, text="Chief Executive Officer: We saw steady demand. Costs improved."),
    ])
    chunks = chunk_transcript(doc, chunk_size=30, overlap=5)
    assert chunks
    assert all(c.page_number == 2 for c in chunks)
    assert all(c.speaker == "Chief Executive Officer" for c in chunks)
    assert all("We saw steady demand" in c.original_text for c in chunks)


def test_chunker_refuses_non_transcript():
    with pytest.raises(ValueError, match="cannot be chunked for transcript RAG"):
        chunk_transcript(make_doc(text="financial text", doc_type=DocumentType.FINANCIAL_REPORT))


def test_embedder_batches_normalizes_and_rejects_bad_vectors():
    embedder = HuggingFaceEmbedder(model=FakeModel(), batch_size=2)
    vectors = embedder.encode(["one", "two"])
    assert vectors.shape == (2, 3)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), [1, 1])
    assert embedder.count_tokens("a few words") == 3
    with pytest.raises(ValueError, match="batch_size"):
        HuggingFaceEmbedder(model=FakeModel(), batch_size=0)

    class BadModel(FakeModel):
        def encode(self, texts, **kwargs):
            return np.zeros((len(texts), 3), dtype=np.float32)

    with pytest.raises(EmbeddingError, match="zero vector"):
        HuggingFaceEmbedder(model=BadModel()).encode(["x"])


def test_faiss_add_duplicate_detection_search_and_persistence(tmp_path):
    store = PersistentFaissStore(tmp_path, dimension=3, embedding_model="synthetic-model")
    chunks = [make_chunk("c1"), make_chunk("c2", page=2)]
    vectors = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    assert store.add(chunks, vectors) == 2
    assert store.add(chunks, vectors) == 0
    # Adding a document containing multiple chunks must not discard its later chunks.
    more = [make_chunk("c3", digest="b" * 64), make_chunk("c4", page=3, digest="b" * 64)]
    assert store.add(more, vectors) == 2
    assert store.size == 4
    hits = store.search(np.array([1, 0, 0]), top_k=2, similarity_threshold=0.8)
    assert hits[0].chunk.chunk_id in {"c1", "c3"}  # Exact-vector ties are implementation-defined.
    assert hits[0].relevant is True
    assert store.search(np.array([1, 0, 0]), document_name="absent.pdf") == []

    generation = store.save()
    assert generation
    loaded = PersistentFaissStore.open(tmp_path, dimension=3, embedding_model="synthetic-model")
    assert loaded.size == 4
    assert loaded.generation == generation
    assert loaded.chunks[0].page_number == 1
    with pytest.raises(EmbeddingModelMismatch):
        PersistentFaissStore.open(tmp_path, dimension=3, embedding_model="another-model")


def test_corrupt_generation_is_rejected_before_faiss_load(tmp_path):
    store = PersistentFaissStore(tmp_path, dimension=3, embedding_model="synthetic-model")
    store.add([make_chunk()], np.array([[1, 0, 0]], dtype=np.float32))
    store.save()
    pointer = json.loads((tmp_path / "CURRENT.json").read_text(encoding="utf-8"))
    index_file = tmp_path / f"transcripts-{pointer['generation']}.faiss"
    index_file.write_bytes(index_file.read_bytes() + b"corruption")
    with pytest.raises(VectorStoreError, match="checksum"):
        PersistentFaissStore.open(tmp_path, dimension=3, embedding_model="synthetic-model")


def test_retriever_returns_no_match_for_empty_store(tmp_path):
    embedder = HuggingFaceEmbedder(model=FakeModel())
    store = PersistentFaissStore(tmp_path, dimension=3, embedding_model="synthetic")
    response = TranscriptRetriever(store, embedder).retrieve("operating margins")
    assert response.matches == []
    assert response.rejected_reason

