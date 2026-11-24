"""Validated models shared by the PDF ingestion pipeline."""

from __future__ import annotations

from enum import Enum
from pydantic import BaseModel, ConfigDict, Field


class DocumentType(str, Enum):
    FINANCIAL_REPORT = "FINANCIAL_REPORT"
    EARNINGS_TRANSCRIPT = "EARNINGS_TRANSCRIPT"
    UNKNOWN = "UNKNOWN"


class ProcessingStatus(str, Enum):
    PROCESSED = "PROCESSED"
    OCR_REQUIRED = "OCR_REQUIRED"
    EMPTY = "EMPTY"
    FAILED = "FAILED"
    DUPLICATE = "DUPLICATE"
    UNKNOWN = "UNKNOWN"


class ExtractedPage(BaseModel):
    page_number: int = Field(ge=1)
    text: str = ""
    has_images: bool = False
    error: str | None = None


class ExtractedTable(BaseModel):
    page_number: int = Field(ge=1)
    table_number: int = Field(ge=1)
    rows: list[list[str | None]]


class DocumentMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")

    original_filename: str
    source_path: str
    sha256: str | None = None
    title: str | None = None
    author: str | None = None
    subject: str | None = None
    creation_date: str | None = None
    page_count: int = Field(default=0, ge=0)
    document_type: DocumentType = DocumentType.UNKNOWN
    company: str | None = None
    reporting_period: str | None = None
    report_period_type: str | None = None
    consolidation: str | None = None
    earnings_call_date: str | None = None
    management_speakers: list[str] = Field(default_factory=list)


class ProcessedDocument(BaseModel):
    metadata: DocumentMetadata
    pages: list[ExtractedPage] = Field(default_factory=list)
    tables: list[ExtractedTable] = Field(default_factory=list)
    processing_status: ProcessingStatus
    errors: list[str] = Field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(page.text for page in self.pages if page.text)


class DocumentProcessingResult(BaseModel):
    source_path: str
    processing_status: ProcessingStatus
    document: ProcessedDocument | None = None
    errors: list[str] = Field(default_factory=list)


class IngestionSummary(BaseModel):
    scanned_directories: list[str] = Field(default_factory=list)
    discovered_pdfs: int = Field(default=0, ge=0)
    processed: int = Field(default=0, ge=0)
    duplicates: int = Field(default=0, ge=0)
    failed: int = Field(default=0, ge=0)
    ocr_required: int = Field(default=0, ge=0)
    unknown: int = Field(default=0, ge=0)
    results: list[DocumentProcessingResult] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
