"""Pydantic schemas for traceable financial statement extraction."""

from __future__ import annotations

from decimal import Decimal
from enum import Enum

from pydantic import BaseModel, Field


class StatementBasis(str, Enum):
    CONSOLIDATED = "CONSOLIDATED"
    STANDALONE = "STANDALONE"
    UNKNOWN = "UNKNOWN"


class MetricStatus(str, Enum):
    EXTRACTED = "EXTRACTED"
    CALCULATED = "CALCULATED"
    NOT_FOUND = "NOT_FOUND"
    REVIEW = "REVIEW"


class ExtractionStatus(str, Enum):
    SUCCESS = "SUCCESS"
    PARTIAL = "PARTIAL"
    REVIEW = "REVIEW"
    FAILED = "FAILED"


class FinancialMetric(BaseModel):
    name: str
    value: Decimal | None = None
    original_value: str | None = None
    unit: str | None = None
    currency: str | None = None
    scale: str | None = None
    reporting_period: str
    comparative_value: Decimal | None = None
    comparative_original_value: str | None = None
    comparative_period: str | None = None
    source_document: str | None = None
    source_page: int | None = Field(default=None, ge=1)
    source_row_label: str | None = None
    statement_basis: StatementBasis | None = None
    extraction_status: MetricStatus
    is_calculated: bool = False
    formula: str | None = None
    validation_errors: list[str] = Field(default_factory=list)


class FinancialStatement(BaseModel):
    company: str = "TCS"
    document_type: str = "FINANCIAL_REPORT"
    source_document: str
    reporting_period: str
    report_period_type: str
    statement_basis: StatementBasis
    currency: str | None = "INR"
    unit: str | None = "CRORES"
    financial_metrics: dict[str, FinancialMetric]
    extraction_status: ExtractionStatus
    validation_errors: list[str] = Field(default_factory=list)
    unrecognized_metrics: list[str] = Field(default_factory=list)


class MetricComparison(BaseModel):
    name: str
    value: Decimal
    unit: str | None = None
    currency: str | None = None
    formula: str
    source_documents: list[str]
    source_pages: list[int]


class QuarterlyComparison(BaseModel):
    from_period: str
    to_period: str
    statement_basis: StatementBasis
    metrics: dict[str, MetricComparison]
    notes: list[str] = Field(default_factory=list)


class FinancialExtractionOutput(BaseModel):
    company: str = "TCS"
    statements: list[FinancialStatement]
    quarterly_comparisons: list[QuarterlyComparison] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
