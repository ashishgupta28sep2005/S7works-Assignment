"""Validated public and LLM-facing schemas for transcript analysis."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SourceReference(StrictModel):
    source_id: str
    document_name: str
    reporting_period: str | None = None
    page_number: int = Field(ge=1)
    speaker: str | None = None
    retrieval_score: float


class RecurringTheme(StrictModel):
    theme: str
    description: str
    supporting_evidence: list[str] = Field(min_length=1)
    source_references: list[SourceReference] = Field(min_length=1)


class ManagementSentiment(StrictModel):
    sentiment: Literal["positive", "neutral", "cautious", "negative", "mixed"]
    explanation: str
    supporting_evidence: list[str] = Field(min_length=1)
    source_references: list[SourceReference] = Field(min_length=1)


class ForwardLookingStatement(StrictModel):
    statement: str
    business_area: str
    outlook_direction: Literal["positive", "neutral", "negative", "uncertain"]
    source_references: list[SourceReference] = Field(min_length=1)


class BusinessRisk(StrictModel):
    risk: str
    potential_business_impact: str
    source_references: list[SourceReference] = Field(min_length=1)


class BusinessOpportunity(StrictModel):
    opportunity: str
    potential_business_benefit: str
    source_references: list[SourceReference] = Field(min_length=1)


class QualitativeAnalysisResult(StrictModel):
    company: str = "TCS"
    analysis_question: str
    reporting_periods_analyzed: list[str] = Field(default_factory=list)
    recurring_themes: list[RecurringTheme] = Field(default_factory=list)
    management_sentiment: list[ManagementSentiment] = Field(default_factory=list)
    forward_looking_statements: list[ForwardLookingStatement] = Field(default_factory=list)
    business_risks: list[BusinessRisk] = Field(default_factory=list)
    business_opportunities: list[BusinessOpportunity] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    sources: list[SourceReference] = Field(default_factory=list)


# The model cites these IDs; the application resolves IDs into trusted references.
class _LLMTheme(StrictModel):
    theme: str
    description: str
    supporting_evidence: list[str] = Field(min_length=1)
    source_references: list[str] = Field(min_length=1)


class _LLMSentiment(StrictModel):
    sentiment: Literal["positive", "neutral", "cautious", "negative", "mixed"]
    explanation: str
    supporting_evidence: list[str] = Field(min_length=1)
    source_references: list[str] = Field(min_length=1)


class _LLMForwardStatement(StrictModel):
    statement: str
    business_area: str
    outlook_direction: Literal["positive", "neutral", "negative", "uncertain"]
    source_references: list[str] = Field(min_length=1)


class _LLMRisk(StrictModel):
    risk: str
    potential_business_impact: str
    source_references: list[str] = Field(min_length=1)


class _LLMOpportunity(StrictModel):
    opportunity: str
    potential_business_benefit: str
    source_references: list[str] = Field(min_length=1)


class LLMQualitativeAnalysis(StrictModel):
    recurring_themes: list[_LLMTheme] = Field(default_factory=list)
    management_sentiment: list[_LLMSentiment] = Field(default_factory=list)
    forward_looking_statements: list[_LLMForwardStatement] = Field(default_factory=list)
    business_risks: list[_LLMRisk] = Field(default_factory=list)
    business_opportunities: list[_LLMOpportunity] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
