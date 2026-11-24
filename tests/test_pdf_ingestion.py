"""Technical ingestion tests use synthetic PDFs, not TCS source documents."""

from pathlib import Path

import pymupdf
import pytest

from app.ingestion.document_classifier import classify_document
from app.ingestion.document_manager import ingest_documents
from app.ingestion.models import (
    DocumentMetadata,
    DocumentType,
    ExtractedPage,
    ProcessedDocument,
    ProcessingStatus,
)
from app.ingestion.pdf_loader import PDFLoadError, load_pdf


def make_pdf(path: Path, page_texts: list[str]) -> Path:
    pdf = pymupdf.open()
    for text in page_texts:
        page = pdf.new_page()
        if text:
            page.insert_text((72, 72), text)
    pdf.save(path)
    pdf.close()
    return path


def make_table_pdf(path: Path) -> Path:
    pdf = pymupdf.open()
    page = pdf.new_page()
    x0, y0, cell_w, cell_h = 72, 72, 140, 30
    values = [["Metric", "Value"], ["Revenue", "100"]]
    for row in range(2):
        for col in range(2):
            rect = pymupdf.Rect(x0 + col * cell_w, y0 + row * cell_h,
                                x0 + (col + 1) * cell_w, y0 + (row + 1) * cell_h)
            page.draw_rect(rect)
            page.insert_text((rect.x0 + 5, rect.y0 + 20), values[row][col])
    pdf.save(path)
    pdf.close()
    return path


def test_pdf_text_and_page_numbers(tmp_path: Path) -> None:
    document = load_pdf(make_pdf(tmp_path / "multi.pdf", ["First page text", "Second page text"]))
    assert [page.page_number for page in document.pages] == [1, 2]
    assert "First page text" in document.pages[0].text
    assert "Second page text" in document.pages[1].text
    assert document.processing_status == ProcessingStatus.PROCESSED


def test_table_extraction(tmp_path: Path) -> None:
    document = load_pdf(make_table_pdf(tmp_path / "table.pdf"))
    assert document.tables
    assert document.tables[0].page_number == 1
    assert any("Revenue" in cell for row in document.tables[0].rows for cell in row if cell)


def test_empty_pdf(tmp_path: Path) -> None:
    document = load_pdf(make_pdf(tmp_path / "empty.pdf", [""]))
    assert document.processing_status == ProcessingStatus.EMPTY
    assert document.pages[0].text == ""


def test_image_only_pdf_is_marked_ocr_required(tmp_path: Path) -> None:
    pdf = pymupdf.open()
    page = pdf.new_page()
    pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 80, 80), 0)
    pixmap.clear_with(220)
    page.insert_image(pymupdf.Rect(50, 50, 130, 130), pixmap=pixmap)
    path = tmp_path / "scan.pdf"
    pdf.save(path)
    pdf.close()
    document = load_pdf(path)
    assert document.processing_status == ProcessingStatus.OCR_REQUIRED
    assert any("OCR is required" in error for error in document.errors)


def test_corrupted_pdf(tmp_path: Path) -> None:
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"not a pdf")
    with pytest.raises(PDFLoadError):
        load_pdf(path)


def test_password_protected_pdf(tmp_path: Path) -> None:
    path = tmp_path / "protected.pdf"
    pdf = pymupdf.open()
    pdf.new_page().insert_text((72, 72), "private test document")
    pdf.save(path, encryption=pymupdf.PDF_ENCRYPT_AES_256,
             user_pw="test-password", owner_pw="owner-password")
    pdf.close()
    with pytest.raises(PDFLoadError, match="Password-protected"):
        load_pdf(path)


def test_unknown_document_classification() -> None:
    document = ProcessedDocument(
        metadata=DocumentMetadata(original_filename="unknown.pdf", source_path="unknown.pdf"),
        pages=[ExtractedPage(page_number=1, text="A general business document with no relevant headings.")],
        processing_status=ProcessingStatus.PROCESSED,
    )
    result = classify_document(document)
    assert result.metadata.document_type == DocumentType.UNKNOWN
    assert result.errors


def test_financial_report_classification() -> None:
    document = ProcessedDocument(
        metadata=DocumentMetadata(original_filename="x.pdf", source_path="x.pdf"),
        pages=[ExtractedPage(page_number=1, text=(
            "Tata Consultancy Services. Interim Financial Results for the quarter ended June 30, 2026. "
            "Consolidated financial results. Statement of profit and loss."
        ))],
        processing_status=ProcessingStatus.PROCESSED,
    )
    result = classify_document(document)
    assert result.metadata.document_type == DocumentType.FINANCIAL_REPORT
    assert result.metadata.company == "TCS"
    assert result.metadata.report_period_type == "QUARTERLY"
    assert result.metadata.consolidation == "CONSOLIDATED"


def test_earnings_transcript_classification() -> None:
    document = ProcessedDocument(
        metadata=DocumentMetadata(original_filename="x.pdf", source_path="x.pdf"),
        pages=[ExtractedPage(page_number=1, text=(
            "TCS Earnings Conference Call Transcript for Q2 FY 2026-27 held October 14, 2026. "
            "Transcript of the earnings conference call. Operator: Welcome. "
            "Mr. Riya Example, Chief Financial Officer."
        ))],
        processing_status=ProcessingStatus.PROCESSED,
    )
    result = classify_document(document)
    assert result.metadata.document_type == DocumentType.EARNINGS_TRANSCRIPT
    assert result.metadata.company == "TCS"
    assert result.metadata.reporting_period == "Q2 FY2026-27"
    assert result.metadata.earnings_call_date == "2026-10-14"
    assert "Riya Example" in result.metadata.management_speakers


def test_transcript_uses_metadata_title_for_period_and_call_date() -> None:
    document = ProcessedDocument(
        metadata=DocumentMetadata(
            original_filename="title.pdf", source_path="title.pdf",
            title="Transcript of the Q4 2025-26 Earnings Conference Call held at 19:00 hrs IST on April 9, 2026",
        ),
        pages=[ExtractedPage(page_number=1, text=(
            "Tata Consultancy Services Limited Earnings Conference Call. "
            "Transcript of the earnings conference call. Operator: Welcome."
        ))],
        processing_status=ProcessingStatus.PROCESSED,
    )
    result = classify_document(document)
    assert result.metadata.document_type == DocumentType.EARNINGS_TRANSCRIPT
    assert result.metadata.reporting_period == "Q4 FY2025-26"
    assert result.metadata.earnings_call_date == "2026-04-09"


def test_duplicate_detection_and_summary(tmp_path: Path) -> None:
    first, second = tmp_path / "a", tmp_path / "b"
    first.mkdir()
    second.mkdir()
    (first / "one.pdf").write_bytes(b"identical bytes")
    (second / "two.pdf").write_bytes(b"identical bytes")
    summary = ingest_documents(first, second)
    assert summary.discovered_pdfs == 2
    assert summary.duplicates == 1
    assert len(summary.results) == 2


def test_missing_document_directories(tmp_path: Path) -> None:
    summary = ingest_documents(tmp_path / "missing1", tmp_path / "missing2")
    assert summary.discovered_pdfs == 0
    assert len(summary.errors) == 3
    assert summary.failed == 0


def test_summary_counts_classified_documents(tmp_path: Path) -> None:
    reports, calls = tmp_path / "reports", tmp_path / "calls"
    reports.mkdir()
    calls.mkdir()
    make_pdf(reports / "report.pdf", [
        "TCS Interim Financial Results for the quarter ended June 30, 2026. "
        "Consolidated financial results. Statement of profit and loss."
    ])
    summary = ingest_documents(reports, calls)
    assert summary.discovered_pdfs == 1
    assert summary.processed == 1
    assert summary.results[0].document.metadata.document_type == DocumentType.FINANCIAL_REPORT
