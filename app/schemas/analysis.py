"""Public result schemas for deterministic financial and qualitative orchestration."""

from __future__ import annotations

from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.financial import FinancialExtractionOutput, StatementBasis
from app.schemas.qualitative import QualitativeAnalysisResult, SourceReference


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FinancialMetricReference(StrictModel):
    source_id: str
    metric_name: str
    reporting_period: str
    report_period_type: Literal["QUARTERLY", "ANNUAL"]
    statement_basis: StatementBasis
    value: Decimal
    comparative_value: Decimal | None = None
    comparative_period: str | None = None
    unit: str | None = None
    currency: str | None = None
    source_document: str
    source_page: int = Field(ge=1)


class CrossAnalysisFinding(StrictModel):
    finding: str
    relationship: Literal["agreement", "divergence", "context", "insufficient_evidence"]
    financial_sources: list[FinancialMetricReference] = Field(default_factory=list)
    transcript_sources: list[SourceReference] = Field(default_factory=list)

    @model_validator(mode="after")
    def avoid_mixed_statement_bases_and_period_kinds(self) -> "CrossAnalysisFinding":
        if self.relationship in {"agreement", "divergence"} and (
            not self.financial_sources or not self.transcript_sources
        ):
            raise ValueError("Agreement/divergence findings require financial and transcript sources")
        bases = {source.statement_basis for source in self.financial_sources}
        period_kinds = {source.report_period_type for source in self.financial_sources}
        if len(bases) > 1:
            raise ValueError("A cross-analysis finding cannot mix consolidated and standalone figures")
        if len(period_kinds) > 1:
            raise ValueError("A cross-analysis finding cannot mix quarterly and annual figures")
        if self.relationship in {"agreement", "divergence"} and StatementBasis.UNKNOWN in bases:
            raise ValueError("Agreement/divergence findings require an identifiable financial statement basis")
        return self


class ToolStatus(StrictModel):
    status: Literal["success", "failed"]
    error: str | None = None


class ToolExecutionStatus(StrictModel):
    financial_extraction: Literal["success", "failed"]
    qualitative_analysis: Literal["success", "failed"]
    master_reasoning: Literal["success", "failed"]
    overall_status: Literal["success", "partial_success", "failed"]
    errors: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def status_matches_tool_results(self) -> "ToolExecutionStatus":
        specialists_ok = self.financial_extraction == "success" or self.qualitative_analysis == "success"
        both_ok = self.financial_extraction == "success" and self.qualitative_analysis == "success"
        expected = "failed" if not specialists_ok else (
            "success" if both_ok and self.master_reasoning == "success" else "partial_success"
        )
        if self.overall_status != expected:
            raise ValueError(f"overall_status must be {expected!r} for the component statuses")
        return self


class FinancialAnalysisResult(StrictModel):
    company: str
    analysis_question: str
    reporting_periods: list[str] = Field(default_factory=list)
    financial_analysis: FinancialExtractionOutput | None = None
    qualitative_analysis: QualitativeAnalysisResult | None = None
    cross_analysis: list[CrossAnalysisFinding] = Field(default_factory=list)
    data_quality_notes: list[str] = Field(default_factory=list)
    tool_execution_status: ToolExecutionStatus


class _LLMCrossAnalysisFinding(StrictModel):
    finding: str
    relationship: Literal["agreement", "divergence", "context", "insufficient_evidence"]
    financial_source_ids: list[str] = Field(default_factory=list)
    transcript_source_ids: list[str] = Field(default_factory=list)


class LLMAnalysisResponse(StrictModel):
    cross_analysis: list[_LLMCrossAnalysisFinding] = Field(default_factory=list)
