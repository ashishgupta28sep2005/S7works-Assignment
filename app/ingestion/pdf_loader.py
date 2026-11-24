"""PDF text, metadata, image presence, and table extraction."""

from __future__ import annotations

import logging
from pathlib import Path

import pdfplumber
import pymupdf

from app.ingestion.models import (
    DocumentMetadata,
    ExtractedPage,
    ExtractedTable,
    ProcessedDocument,
    ProcessingStatus,
)

logger = logging.getLogger(__name__)


class PDFLoadError(Exception):
    """Raised when a PDF cannot be opened or read at all."""


def load_pdf(path: str | Path) -> ProcessedDocument:
    """Extract real page text and tables; scanned pages are marked OCR_REQUIRED."""
    pdf_path = Path(path)
    if not pdf_path.is_file():
        raise PDFLoadError(f"PDF does not exist or is not a file: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise PDFLoadError(f"Expected a .pdf file: {pdf_path}")

    pages: list[ExtractedPage] = []
    tables: list[ExtractedTable] = []
    errors: list[str] = []
    try:
        with pymupdf.open(pdf_path) as pdf:
            if pdf.is_encrypted:
                raise PDFLoadError(f"Password-protected PDF cannot be opened: {pdf_path.name}")
            page_count = pdf.page_count
            raw_metadata = pdf.metadata or {}
            for index in range(page_count):
                try:
                    page = pdf.load_page(index)
                    text = page.get_text("text").strip()
                    image_list = page.get_images(full=True)
                    pages.append(ExtractedPage(
                        page_number=index + 1,
                        text=text,
                        has_images=bool(image_list),
                    ))
                except Exception as exc:  # A damaged page should not discard other pages.
                    message = f"Page {index + 1} text extraction failed: {exc}"
                    logger.warning("%s (%s)", message, pdf_path)
                    errors.append(message)
                    pages.append(ExtractedPage(page_number=index + 1, error=str(exc)))
    except PDFLoadError:
        raise
    except Exception as exc:
        logger.exception("Unable to open PDF %s", pdf_path)
        raise PDFLoadError(f"Unable to read PDF {pdf_path.name}: {exc}") from exc

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for index, page in enumerate(pdf.pages, start=1):
                try:
                    for table_index, table in enumerate(page.extract_tables() or [], start=1):
                        cleaned = [[_clean_cell(cell) for cell in row] for row in table]
                        if any(any(cell for cell in row) for row in cleaned):
                            tables.append(ExtractedTable(
                                page_number=index,
                                table_number=table_index,
                                rows=cleaned,
                            ))
                except Exception as exc:
                    message = f"Page {index} table extraction failed: {exc}"
                    logger.warning("%s (%s)", message, pdf_path)
                    errors.append(message)
    except Exception as exc:
        message = f"Table extraction failed: {exc}"
        logger.warning("%s (%s)", message, pdf_path)
        errors.append(message)

    nonempty_text_pages = sum(bool(page.text.strip()) for page in pages)
    has_images = any(page.has_images for page in pages)
    if page_count == 0:
        status = ProcessingStatus.EMPTY
    elif has_images and nonempty_text_pages / page_count < 0.25:
        status = ProcessingStatus.OCR_REQUIRED
        errors.append("PDF appears scanned or image-based; OCR is required to read image text.")
    elif not nonempty_text_pages and not tables:
        status = ProcessingStatus.EMPTY
        errors.append("PDF contains no extractable text or tables.")
    else:
        status = ProcessingStatus.PROCESSED

    metadata = DocumentMetadata(
        original_filename=pdf_path.name,
        source_path=str(pdf_path.resolve()),
        title=_optional(raw_metadata.get("title")),
        author=_optional(raw_metadata.get("author")),
        subject=_optional(raw_metadata.get("subject")),
        creation_date=_optional(raw_metadata.get("creationDate")),
        page_count=page_count,
    )
    return ProcessedDocument(
        metadata=metadata,
        pages=pages,
        tables=tables,
        processing_status=status,
        errors=errors,
    )


def _optional(value: object) -> str | None:
    text = str(value).strip() if value else ""
    return text or None


def _clean_cell(value: object) -> str | None:
    text = " ".join(str(value).split()) if value is not None else ""
    return text or None
