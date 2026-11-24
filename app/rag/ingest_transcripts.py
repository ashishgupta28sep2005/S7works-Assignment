"""Index available earnings-call transcripts into the persistent local FAISS store."""

from __future__ import annotations

import argparse
import logging

from app.config.settings import get_settings
from app.ingestion.document_manager import ingest_documents
from app.ingestion.models import DocumentType, ProcessingStatus
from app.rag.embeddings import HuggingFaceEmbedder
from app.rag.text_chunker import chunk_transcript
from app.rag.vector_store import PersistentFaissStore

logger = logging.getLogger(__name__)


def index_transcripts(*, rebuild: bool = False) -> tuple[PersistentFaissStore, int, list[str]]:
    settings = get_settings()
    summary = ingest_documents()
    documents = [
        result.document for result in summary.results
        if result.document is not None
        and result.processing_status == ProcessingStatus.PROCESSED
        and result.document.metadata.document_type == DocumentType.EARNINGS_TRANSCRIPT
    ]
    documents.sort(key=lambda doc: (doc.metadata.reporting_period or "", doc.metadata.original_filename))
    filenames = [doc.metadata.original_filename for doc in documents]
    if len(documents) == 2:
        logger.warning("Only two valid earnings transcripts are available; a third historical transcript is missing.")
    elif len(documents) < 2:
        logger.warning("Only %d valid earnings transcript(s) are available.", len(documents))

    embedder = HuggingFaceEmbedder(
        model_name=settings.embedding_model,
        device=settings.embedding_device,
        batch_size=settings.embedding_batch_size,
    )
    store = PersistentFaissStore.open(
        settings.vectorstore_dir,
        dimension=embedder.dimension,
        embedding_model=settings.embedding_model,
        rebuild=rebuild,
    )
    added = 0
    for document in documents:
        digest = document.metadata.sha256
        if not digest:
            raise RuntimeError(f"Document manager did not supply a content hash for {document.metadata.original_filename}")
        if store.has_document(digest):
            logger.info("Already indexed; skipping %s", document.metadata.original_filename)
            continue
        chunks = chunk_transcript(
            document,
            chunk_size=settings.rag_chunk_size,
            overlap=settings.rag_chunk_overlap,
            token_counter=embedder.count_tokens,
        )
        if not chunks:
            logger.warning("No non-empty transcript chunks produced for %s", document.metadata.original_filename)
            continue
        vectors = embedder.encode([chunk.original_text for chunk in chunks])
        added += store.add(chunks, vectors)
        logger.info("Prepared %s: %d page-grounded chunks", document.metadata.original_filename, len(chunks))

    if added:
        store.save()
    elif rebuild and store.size:
        # A rebuild with no new sources must not replace an existing valid generation.
        logger.warning("Rebuild produced no chunks; no new generation was committed.")
    return store, added, filenames


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Build a fresh generation from all currently available transcripts.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        store, added, filenames = index_transcripts(rebuild=args.rebuild)
    except Exception as exc:
        logger.exception("Transcript vector indexing failed")
        print(f"Indexing failed: {exc}")
        return 1
    print(f"Valid transcripts: {len(filenames)} ({', '.join(filenames) if filenames else 'none'})")
    print(f"New chunks indexed: {added}; total vectors: {store.size}")
    print(f"Vector store: {get_settings().vectorstore_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
