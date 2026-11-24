"""Pydantic schemas for qualitative, source-grounded quarterly outlooks."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.analysis import FinancialMetricReference
from app.schemas.qualitative import SourceReference

OutlookDirection = Literal["positive", "stable", "cautious", "negative", "uncertain"]
EvidenceStatus = Literal["supported", "limited", "conflicting", "unavailable"]
ConfidenceLevel = Literal["high", "medium", "low"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ForecastSourceReference(StrictModel):
    """A reference resolved from Phase 6 outputs, with optional verbatim evidence."""

    source: FinancialMetricReference | SourceReference
    supporting_evidence: list[str] = Field(default_factory=list)


class ForecastAssessment(StrictModel):
    direction: OutlookDirection
    evidence_status: EvidenceStatus
    outlook: str
    source_references: list[ForecastSourceReference] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_sources_for_supported_claims(self) -> "ForecastAssessment":
        if self.evidence_status == "unavailable":
            if self.direction != "uncertain":
                raise ValueError("Unavailable evidence requires an uncertain outlook direction")
            if self.source_references:
                raise ValueError("Unavailable sections cannot cite evidence they do not have")
        elif not self.source_references:
            raise ValueError("A supported, limited, or conflicting outlook requires source references")
        return self


class ForecastItem(StrictModel):
    item: str
    reasoning: str
    source_references: list[ForecastSourceReference] = Field(min_length=1)


class ForecastReasoning(StrictModel):
    inference: str
    source_references: list[ForecastSourceReference] = Field(min_length=1)


class QuarterlyForecast(StrictModel):
    company: str
    forecast_for: str
    based_on_reporting_periods: list[str] = Field(min_length=1)
    overall_business_outlook: ForecastAssessment
    revenue_outlook: ForecastAssessment
    profitability_outlook: ForecastAssessment
    operating_margin_outlook: ForecastAssessment
    management_outlook: ForecastAssessment
    business_risks: list[ForecastItem] = Field(default_factory=list)
    business_opportunities: list[ForecastItem] = Field(default_factory=list)
    key_growth_drivers: list[ForecastItem] = Field(default_factory=list)
    key_monitoring_indicators: list[ForecastItem] = Field(default_factory=list)
    confidence_level: ConfidenceLevel
    reasoning: list[ForecastReasoning] = Field(default_factory=list)
    source_references: list[ForecastSourceReference] = Field(default_factory=list)
    data_limitations: list[str] = Field(default_factory=list)


class _LLMForecastAssessment(StrictModel):
    direction: OutlookDirection
    evidence_status: EvidenceStatus
    outlook: str
    source_references: list[str] = Field(default_factory=list)


class _LLMForecastItem(StrictModel):
    item: str
    reasoning: str
    source_references: list[str] = Field(min_length=1)


class _LLMForecastReasoning(StrictModel):
    inference: str
    source_references: list[str] = Field(min_length=1)


class LLMQuarterlyForecast(StrictModel):
    """Model response uses citation IDs only; trusted metadata is added by the app."""

    overall_business_outlook: _LLMForecastAssessment
    revenue_outlook: _LLMForecastAssessment
    profitability_outlook: _LLMForecastAssessment
    operating_margin_outlook: _LLMForecastAssessment
    management_outlook: _LLMForecastAssessment
    business_risks: list[_LLMForecastItem] = Field(default_factory=list)
    business_opportunities: list[_LLMForecastItem] = Field(default_factory=list)
    key_growth_drivers: list[_LLMForecastItem] = Field(default_factory=list)
    key_monitoring_indicators: list[_LLMForecastItem] = Field(default_factory=list)
    confidence_level: ConfidenceLevel
    reasoning: list[_LLMForecastReasoning] = Field(default_factory=list)
