"""Persistent FAISS IndexFlatIP store with validated, generation-based snapshots."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from app.rag.models import CurrentIndexPointer, IndexSnapshot, RetrievalHit, TranscriptChunk

logger = logging.getLogger(__name__)
_GENERATION = re.compile(r"^[0-9a-f]{32}$")


class VectorStoreError(RuntimeError):
    """Raised for invalid, mismatched, or corrupt persisted vector data."""


class EmbeddingModelMismatch(VectorStoreError):
    """Raised when an index was built with another embedding model or dimension."""


class PersistentFaissStore:
    """Cosine-similarity index where row i always corresponds to metadata[i].

    Each save writes immutable generation files and atomically changes CURRENT.json
    only after both files have been written and checksummed. Previous generations
    remain available as recovery points.
    """

    pointer_name = "CURRENT.json"

    def __init__(self, directory: str | Path, *, dimension: int, embedding_model: str) -> None:
        if dimension < 1:
            raise ValueError("Embedding dimension must be positive")
        self.directory = Path(directory)
        self.dimension = int(dimension)
        self.embedding_model = embedding_model
        self.index = faiss.IndexFlatIP(self.dimension)
        self.chunks: list[TranscriptChunk] = []
        self.document_hashes: set[str] = set()
        self.generation: str | None = None

    @classmethod
    def open(
        cls,
        directory: str | Path,
        *,
        dimension: int,
        embedding_model: str,
        rebuild: bool = False,
    ) -> "PersistentFaissStore":
        store = cls(directory, dimension=dimension, embedding_model=embedding_model)
        if rebuild:
            return store
        pointer_path = store.directory / cls.pointer_name
        if not pointer_path.exists():
            if any(store.directory.glob("transcripts-*.faiss")) or any(store.directory.glob("metadata-*.json")):
                raise VectorStoreError("Vector-store generation files exist without a valid CURRENT.json pointer")
            return store
        if pointer_path.is_symlink():
            raise VectorStoreError("Refusing to load a symbolic-link vector-store pointer")
        try:
            pointer = CurrentIndexPointer.model_validate_json(pointer_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise VectorStoreError(f"Invalid vector-store pointer: {exc}") from exc

        index_path = store.directory / f"transcripts-{pointer.generation}.faiss"
        metadata_path = store.directory / f"metadata-{pointer.generation}.json"
        if index_path.is_symlink() or metadata_path.is_symlink():
            raise VectorStoreError("Refusing to load symbolic-link vector-store generation files")
        if not index_path.is_file() or not metadata_path.is_file():
            raise VectorStoreError("Current vector-store generation is incomplete")
        index_bytes = index_path.read_bytes()
        metadata_bytes = metadata_path.read_bytes()
        if _sha256(index_bytes) != pointer.index_sha256:
            raise VectorStoreError("FAISS index checksum does not match CURRENT.json")
        if _sha256(metadata_bytes) != pointer.metadata_sha256:
            raise VectorStoreError("Metadata checksum does not match CURRENT.json")
        try:
            snapshot = IndexSnapshot.model_validate_json(metadata_bytes)
        except Exception as exc:
            raise VectorStoreError(f"Invalid vector-store metadata: {exc}") from exc
        if snapshot.generation != pointer.generation:
            raise VectorStoreError("Generation identifier mismatch between pointer and metadata")
        if snapshot.embedding_model != embedding_model or snapshot.dimension != dimension:
            raise EmbeddingModelMismatch(
                f"Stored index uses {snapshot.embedding_model!r} ({snapshot.dimension} dimensions); "
                f"requested {embedding_model!r} ({dimension} dimensions). Use an explicit rebuild."
            )
        if len(snapshot.chunks) != len({chunk.chunk_id for chunk in snapshot.chunks}):
            raise VectorStoreError("Duplicate chunk identifiers in persisted metadata")
        grouped = _group_chunk_ids(snapshot.chunks)
        if grouped != snapshot.document_hashes:
            raise VectorStoreError("Document hash mapping does not match chunk metadata")

        try:
            index = faiss.read_index(str(index_path))
        except Exception as exc:
            raise VectorStoreError(f"Could not read validated FAISS index: {exc}") from exc
        if not isinstance(index, faiss.IndexFlatIP):
            raise VectorStoreError("Unexpected FAISS index type; expected IndexFlatIP")
        if index.d != snapshot.dimension or index.ntotal != len(snapshot.chunks):
            raise VectorStoreError("FAISS dimension/count does not match the metadata snapshot")
        store.index = index
        store.chunks = snapshot.chunks
        store.document_hashes = set(snapshot.document_hashes)
        store.generation = snapshot.generation
        return store

    @property
    def size(self) -> int:
        return int(self.index.ntotal)

    def has_document(self, document_hash: str) -> bool:
        return document_hash in self.document_hashes

    def add(self, chunks: list[TranscriptChunk], embeddings: np.ndarray) -> int:
        vectors = np.asarray(embeddings, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape != (len(chunks), self.dimension):
            raise VectorStoreError(
                f"Expected vector matrix {(len(chunks), self.dimension)}, got {vectors.shape}"
            )
        if not np.isfinite(vectors).all():
            raise VectorStoreError("Cannot index NaN or infinite vector values")
        if len(chunks) == 0:
            return 0
        norms = np.linalg.norm(vectors, axis=1)
        if np.any(norms <= 1e-12):
            raise VectorStoreError("Cannot index zero-length embedding vectors")

        selected: list[int] = []
        seen_ids = {chunk.chunk_id for chunk in self.chunks}
        existing_docs = set(self.document_hashes)
        for idx, chunk in enumerate(chunks):
            if chunk.document_hash in existing_docs or chunk.chunk_id in seen_ids:
                continue
            selected.append(idx)
            seen_ids.add(chunk.chunk_id)
        if not selected:
            return 0
        selected_vectors = np.ascontiguousarray(vectors[selected], dtype=np.float32)
        faiss.normalize_L2(selected_vectors)
        self.index.add(selected_vectors)
        for idx in selected:
            self.chunks.append(chunks[idx])
            self.document_hashes.add(chunks[idx].document_hash)
        return len(selected)

    def search(
        self,
        query_embedding: np.ndarray,
        *,
        top_k: int = 5,
        similarity_threshold: float = 0.25,
        reporting_period: str | None = None,
        document_name: str | None = None,
    ) -> list[RetrievalHit]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if not -1.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between -1 and 1")
        query = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        if query.shape[1] != self.dimension:
            raise VectorStoreError(f"Query dimension {query.shape[1]} does not match {self.dimension}")
        if not np.isfinite(query).all() or np.linalg.norm(query) <= 1e-12:
            raise VectorStoreError("Query embedding must be finite and non-zero")
        if self.size == 0:
            return []
        faiss.normalize_L2(query)
        scores, indices = self.index.search(query, self.size)
        hits: list[RetrievalHit] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = self.chunks[int(idx)]
            if reporting_period and chunk.reporting_period != reporting_period:
                continue
            if document_name and chunk.document_name != document_name:
                continue
            hits.append(RetrievalHit(
                score=float(score), relevant=float(score) >= similarity_threshold, chunk=chunk,
            ))
            if len(hits) >= top_k:
                break
        return hits

    def save(self) -> str | None:
        if self.size == 0:
            logger.info("Skipping persistence of an empty FAISS index")
            return None
        self.directory.mkdir(parents=True, exist_ok=True)
        generation = uuid.uuid4().hex
        index_name = f"transcripts-{generation}.faiss"
        metadata_name = f"metadata-{generation}.json"
        index_path = self.directory / index_name
        metadata_path = self.directory / metadata_name
        index_temp = self.directory / f".{index_name}.tmp"
        metadata_temp = self.directory / f".{metadata_name}.tmp"
        pointer_temp = self.directory / f".{self.pointer_name}.{generation}.tmp"
        try:
            faiss.write_index(self.index, str(index_temp))
            index_bytes = index_temp.read_bytes()
            snapshot = IndexSnapshot(
                generation=generation,
                embedding_model=self.embedding_model,
                dimension=self.dimension,
                chunks=self.chunks,
                document_hashes=_group_chunk_ids(self.chunks),
            )
            metadata_bytes = snapshot.model_dump_json(indent=2).encode("utf-8")
            metadata_temp.write_bytes(metadata_bytes)
            os.replace(index_temp, index_path)
            os.replace(metadata_temp, metadata_path)
            pointer = CurrentIndexPointer(
                generation=generation,
                index_sha256=_sha256(index_bytes),
                metadata_sha256=_sha256(metadata_bytes),
            )
            pointer_temp.write_text(pointer.model_dump_json(indent=2), encoding="utf-8")
            os.replace(pointer_temp, self.directory / self.pointer_name)
            self.generation = generation
        except Exception:
            for temp_path in (index_temp, metadata_temp, pointer_temp):
                temp_path.unlink(missing_ok=True)
            raise
        return generation


def _group_chunk_ids(chunks: list[TranscriptChunk]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for chunk in chunks:
        grouped.setdefault(chunk.document_hash, []).append(chunk.chunk_id)
    return grouped


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
