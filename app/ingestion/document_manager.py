"""Directory-level PDF ingestion with duplicate detection and CLI summary."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from app.ingestion.document_classifier import classify_document
from app.ingestion.models import (
    DocumentProcessingResult,
    DocumentType,
    IngestionSummary,
    ProcessingStatus,
)
from app.ingestion.pdf_loader import PDFLoadError, load_pdf

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def ingest_documents(
    financial_reports_dir: str | Path | None = None,
    earnings_transcripts_dir: str | Path | None = None,
) -> IngestionSummary:
    """Process every PDF in configured directories, continuing after failures."""
    directories = [
        Path(financial_reports_dir) if financial_reports_dir is not None else PROJECT_ROOT / "data" / "financial_reports",
        Path(earnings_transcripts_dir) if earnings_transcripts_dir is not None else PROJECT_ROOT / "data" / "earnings_transcripts",
    ]
    summary = IngestionSummary(scanned_directories=[str(path) for path in directories])
    pdf_paths: list[Path] = []
    for directory in directories:
        if not directory.is_dir():
            message = f"Document directory is missing: {directory}"
            logger.warning(message)
            summary.errors.append(message)
            continue
        pdf_paths.extend(sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() == ".pdf"))

    summary.discovered_pdfs = len(pdf_paths)
    if not pdf_paths:
        message = "No PDF documents were found in the configured document directories."
        logger.warning(message)
        summary.errors.append(message)
    seen: dict[str, Path] = {}
    for path in pdf_paths:
        try:
            digest = _sha256(path)
            if digest in seen:
                message = f"Duplicate PDF content; first seen at {seen[digest]}"
                logger.info("Skipping duplicate %s: %s", path, message)
                summary.duplicates += 1
                summary.results.append(DocumentProcessingResult(
                    source_path=str(path), processing_status=ProcessingStatus.DUPLICATE, errors=[message],
                ))
                continue
            seen[digest] = path
            document = classify_document(load_pdf(path))
            document.metadata.sha256 = digest
            if document.metadata.document_type == DocumentType.UNKNOWN:
                document.processing_status = ProcessingStatus.UNKNOWN
                summary.unknown += 1
            elif document.processing_status == ProcessingStatus.OCR_REQUIRED:
                summary.ocr_required += 1
            elif document.processing_status == ProcessingStatus.EMPTY:
                summary.failed += 1
            else:
                summary.processed += 1
            summary.results.append(DocumentProcessingResult(
                source_path=str(path), processing_status=document.processing_status, document=document,
                errors=list(document.errors),
            ))
        except (OSError, PDFLoadError, ValueError) as exc:
            message = str(exc)
            logger.exception("Failed to ingest %s", path)
            summary.failed += 1
            summary.results.append(DocumentProcessingResult(
                source_path=str(path), processing_status=ProcessingStatus.FAILED, errors=[message],
            ))
        except Exception as exc:  # Contain unexpected errors per file.
            logger.exception("Unexpected ingestion error for %s", path)
            summary.failed += 1
            summary.results.append(DocumentProcessingResult(
                source_path=str(path), processing_status=ProcessingStatus.FAILED,
                errors=[f"Unexpected error: {exc}"],
            ))
    return summary


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    summary = ingest_documents()
    output = {
        "discovered_pdfs": summary.discovered_pdfs,
        "processed": summary.processed,
        "duplicates": summary.duplicates,
        "failed": summary.failed,
        "ocr_required": summary.ocr_required,
        "unknown": summary.unknown,
        "errors": summary.errors,
        "documents": [
            {
                "filename": result.document.metadata.original_filename if result.document else Path(result.source_path).name,
                "type": result.document.metadata.document_type.value if result.document else None,
                "status": result.processing_status.value,
                "company": result.document.metadata.company if result.document else None,
                "period": result.document.metadata.reporting_period if result.document else None,
                "report_period_type": result.document.metadata.report_period_type if result.document else None,
                "consolidation": result.document.metadata.consolidation if result.document else None,
                "earnings_call_date": result.document.metadata.earnings_call_date if result.document else None,
                "management_speakers": result.document.metadata.management_speakers if result.document else [],
                "pages": len(result.document.pages) if result.document else 0,
                "tables": len(result.document.tables) if result.document else 0,
                "errors": result.errors,
            }
            for result in summary.results
        ],
    }
    print(json.dumps(output, indent=2))
    return 1 if summary.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
