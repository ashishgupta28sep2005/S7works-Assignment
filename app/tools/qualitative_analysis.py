"""Evidence-grounded qualitative analysis over retrieved earnings transcripts."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from typing import Any

from pydantic import ValidationError

from app.config.llm import GroqConfigurationError, GroqRequestError, create_groq_client, invoke_structured
from app.config.settings import get_settings
from app.prompts.qualitative_analysis_prompt import (
    QUALITATIVE_ANALYSIS_SYSTEM_PROMPT,
    render_evidence,
)
from app.rag.models import RetrievalResponse, TranscriptChunk
from app.schemas.qualitative import (
    BusinessOpportunity,
    BusinessRisk,
    ForwardLookingStatement,
    LLMQualitativeAnalysis,
    ManagementSentiment,
    QualitativeAnalysisResult,
    RecurringTheme,
    SourceReference,
)

logger = logging.getLogger(__name__)
DEFAULT_REPORTING_PERIODS = ("Q4 FY2025-26", "Q2 FY27")
_NUMBER = re.compile(r"(?<!\w)\d[\d,]*(?:\.\d+)?%?")


class QualitativeAnalysisError(RuntimeError):
    """Base exception for setup or analysis failures."""

    def __init__(self, message: str, *, category: str | None = None) -> None:
        self.category = category
        super().__init__(message)


class LLMConfigurationError(QualitativeAnalysisError):
    """Raised when provider configuration is missing or unsupported."""


class InvalidSourceReferenceError(QualitativeAnalysisError):
    """Raised when a generated claim cites unavailable or mismatched evidence."""


class QualitativeAnalysisTool:
    """Retrieve transcript evidence and convert structured LLM output to sourced claims.

    Pass an LLM-like object in tests to avoid provider construction and network calls.
    Production callers may omit it to create the configured LangChain Groq client.
    """

    def __init__(self, retriever: Any, llm: Any | None = None, *, api_key: str | None = None) -> None:
        self.retriever = retriever
        self.llm = llm if llm is not None else _create_llm(api_key=api_key)
        self.retrieval_top_k = max(10, int(getattr(retriever, "top_k", 5)) * 3)

    def analyze(
        self,
        analysis_question: str,
        *,
        reporting_periods: Sequence[str] | None = None,
    ) -> QualitativeAnalysisResult:
        question = analysis_question.strip()
        if not question:
            raise ValueError("analysis_question cannot be empty")
        periods = tuple(dict.fromkeys(reporting_periods or DEFAULT_REPORTING_PERIODS))
        retrieved = self._retrieve(question, periods)
        if not retrieved:
            return QualitativeAnalysisResult(
                company="TCS",
                analysis_question=question,
                limitations=["No relevant transcript passages were retrieved; no analysis was generated."],
            )

        chunks = [chunk for chunk, _score in retrieved]
        sources = self._source_map(retrieved)
        source_rows = [
            {
                "source_id": source_id,
                "document_name": chunk.document_name,
                "reporting_period": chunk.reporting_period,
                "page_number": chunk.page_number,
                "speaker": chunk.speaker,
                "speaker_role": chunk.speaker_role,
                "retrieval_score": score,
                "text": chunk.original_text,
            }
            for source_id, (chunk, score) in sources.items()
        ]
        payload = (
            f"Analysis question: {question}\n"
            f"Retrieved passages (the source ID is the only allowed citation token):\n\n"
            f"{render_evidence(source_rows)}"
        )
        messages = [
            ("system", QUALITATIVE_ANALYSIS_SYSTEM_PROMPT),
            ("human", payload),
        ]
        try:
            raw = invoke_structured(self.llm, LLMQualitativeAnalysis, messages)
        except GroqRequestError as exc:
            raise QualitativeAnalysisError(str(exc), category=exc.category) from None
        try:
            response = _parse_llm_response(raw)
        except (ValidationError, ValueError, TypeError):
            raise QualitativeAnalysisError(
                "Groq returned a response that did not satisfy the required structured schema.",
                category="invalid_response",
            ) from None

        self._validate_claims(response, sources)
        response, removed_quotes = _drop_unverifiable_quote_claims(response, sources)
        periods_found = sorted({chunk.reporting_period for chunk in chunks if chunk.reporting_period})
        limitations = list(response.limitations)
        if removed_quotes:
            limitations.append(
                f"Omitted {removed_quotes} claim(s) because its supporting quotation did not exactly match cited text."
            )
        themes: list[RecurringTheme] = []
        for theme in response.recurring_themes:
            refs = self._resolve_refs(theme.source_references, sources)
            cited_periods = {ref.reporting_period for ref in refs if ref.reporting_period}
            if len(cited_periods) < 2:
                limitations.append(
                    f"Omitted purported recurring theme {theme.theme!r}: evidence did not span two reporting periods."
                )
                continue
            self._validate_quotes(theme.supporting_evidence, refs, sources)
            themes.append(RecurringTheme(
                theme=theme.theme,
                description=theme.description,
                supporting_evidence=theme.supporting_evidence,
                source_references=refs,
            ))

        sentiments = []
        for item in response.management_sentiment:
            refs = self._resolve_refs(item.source_references, sources)
            self._validate_quotes(item.supporting_evidence, refs, sources)
            sentiments.append(ManagementSentiment(
                sentiment=item.sentiment, explanation=item.explanation,
                supporting_evidence=item.supporting_evidence, source_references=refs,
            ))
        forward = [ForwardLookingStatement(
            statement=item.statement, business_area=item.business_area,
            outlook_direction=item.outlook_direction,
            source_references=self._resolve_refs(item.source_references, sources),
        ) for item in response.forward_looking_statements]
        risks = [BusinessRisk(
            risk=item.risk, potential_business_impact=item.potential_business_impact,
            source_references=self._resolve_refs(item.source_references, sources),
        ) for item in response.business_risks]
        opportunities = [BusinessOpportunity(
            opportunity=item.opportunity, potential_business_benefit=item.potential_business_benefit,
            source_references=self._resolve_refs(item.source_references, sources),
        ) for item in response.business_opportunities]

        used_refs = _unique_refs(
            [ref for group in themes for ref in group.source_references]
            + [ref for group in sentiments for ref in group.source_references]
            + [ref for group in forward for ref in group.source_references]
            + [ref for group in risks for ref in group.source_references]
            + [ref for group in opportunities for ref in group.source_references]
        )
        return QualitativeAnalysisResult(
            company=chunks[0].company or "TCS",
            analysis_question=question,
            reporting_periods_analyzed=periods_found,
            recurring_themes=themes,
            management_sentiment=sentiments,
            forward_looking_statements=forward,
            business_risks=risks,
            business_opportunities=opportunities,
            limitations=list(dict.fromkeys(limitations)),
            sources=used_refs,
        )

    def _retrieve(self, question: str, periods: Sequence[str]) -> list[tuple[TranscriptChunk, float]]:
        found: list[tuple[TranscriptChunk, float]] = []
        seen: set[str] = set()
        for period in periods:
            result: RetrievalResponse = self.retriever.retrieve(
                question, reporting_period=period, top_k=self.retrieval_top_k,
            )
            for hit in result.matches:
                if not hit.relevant or hit.chunk.speaker_role != "management" or hit.chunk.chunk_id in seen:
                    continue
                found.append((hit.chunk, hit.score))
                seen.add(hit.chunk.chunk_id)
        return found

    @staticmethod
    def _source_map(
        retrieved: Sequence[tuple[TranscriptChunk, float]],
    ) -> dict[str, tuple[TranscriptChunk, float]]:
        return {f"S{index:03d}": row for index, row in enumerate(retrieved, 1)}

    @staticmethod
    def _resolve_refs(
        source_ids: Sequence[str],
        sources: dict[str, tuple[TranscriptChunk, float]],
    ) -> list[SourceReference]:
        if not source_ids:
            raise InvalidSourceReferenceError("Every material claim must cite at least one retrieved source.")
        refs: list[SourceReference] = []
        for source_id in dict.fromkeys(source_ids):
            if source_id not in sources:
                raise InvalidSourceReferenceError(f"Unsupported source reference: {source_id!r}")
            chunk, score = sources[source_id]
            refs.append(SourceReference(
                source_id=source_id,
                document_name=chunk.document_name,
                reporting_period=chunk.reporting_period,
                page_number=chunk.page_number,
                speaker=chunk.speaker,
                retrieval_score=score,
            ))
        return refs

    @staticmethod
    def _validate_quotes(
        quotes: Sequence[str], refs: Sequence[SourceReference],
        sources: dict[str, tuple[TranscriptChunk, float]],
    ) -> None:
        for quote in quotes:
            if not quote.strip() or not any(
                _normalize(quote) in _normalize(sources[ref.source_id][0].original_text)
                for ref in refs
            ):
                raise InvalidSourceReferenceError(
                    "Supporting evidence must be a faithful excerpt from a cited retrieved passage."
                )

    @staticmethod
    def _validate_claims(
        response: LLMQualitativeAnalysis,
        sources: dict[str, tuple[TranscriptChunk, float]],
    ) -> None:
        groups: list[tuple[str, Sequence[str], Sequence[str]]] = []
        groups.extend((item.theme + " " + item.description, item.source_references, item.supporting_evidence)
                      for item in response.recurring_themes)
        groups.extend((item.explanation, item.source_references, item.supporting_evidence)
                      for item in response.management_sentiment)
        groups.extend((item.statement + " " + item.business_area, item.source_references, ())
                      for item in response.forward_looking_statements)
        groups.extend((item.risk + " " + item.potential_business_impact, item.source_references, ())
                      for item in response.business_risks)
        groups.extend((item.opportunity + " " + item.potential_business_benefit,
                       item.source_references, ()) for item in response.business_opportunities)
        for claim, source_ids, quotes in groups:
            invalid = set(source_ids) - sources.keys()
            if invalid:
                raise InvalidSourceReferenceError(f"Unsupported source reference(s): {sorted(invalid)}")
            if not source_ids:
                raise InvalidSourceReferenceError("Every material claim must cite at least one retrieved source.")
            claim_periods = {sources[sid][0].reporting_period for sid in source_ids}
            for number in _NUMBER.findall(claim):
                normalized = number.replace(",", "")
                if not any(
                    normalized in re.sub(r"[,\s]", "", _trusted_numeric_context(sources[sid][0]))
                    for sid in source_ids
                ):
                    raise InvalidSourceReferenceError(
                        f"Numerical claim {number!r} is not present in its cited transcript passage."
                    )
            if None in claim_periods:
                raise InvalidSourceReferenceError("Cited transcript passage is missing its reporting period.")


def _create_llm(*, api_key: str | None = None) -> Any:
    settings = get_settings()
    if settings.llm_provider != "groq":
        raise LLMConfigurationError(
            f"Unsupported LLM_PROVIDER {settings.llm_provider!r}; supported provider is 'groq'."
        )
    key = api_key or settings.groq_api_key
    try:
        if api_key:
            settings = settings.model_copy(update={"groq_api_key": api_key}) if hasattr(settings, "model_copy") else _settings_with_key(settings, api_key)
        return create_groq_client(settings)
    except GroqConfigurationError as exc:
        raise LLMConfigurationError(str(exc)) from None


def _settings_with_key(settings: Any, api_key: str) -> Any:
    """Copy dataclass settings without including the secret in its repr."""
    from dataclasses import replace
    return replace(settings, groq_api_key=api_key)


def _parse_llm_response(raw: Any) -> LLMQualitativeAnalysis:
    if isinstance(raw, LLMQualitativeAnalysis):
        return raw
    if hasattr(raw, "content"):
        raw = raw.content
    if isinstance(raw, list):
        raw = "".join(block.get("text", "") if isinstance(block, dict) else str(block) for block in raw)
    if isinstance(raw, str):
        # Accommodate a provider that wraps valid JSON in a Markdown fence, but no other prose.
        raw = raw.strip()
        if raw.startswith("```json") and raw.endswith("```"):
            raw = raw[7:-3].strip()
        elif raw.startswith("```") and raw.endswith("```"):
            raw = raw[3:-3].strip()
        raw = json.loads(raw)
    return LLMQualitativeAnalysis.model_validate(raw)


def _normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def _trusted_numeric_context(chunk: TranscriptChunk) -> str:
    """Include transcript text and ingestion metadata, never LLM-supplied metadata."""
    metadata = " ".join(value for value in (
        chunk.reporting_period or "", chunk.earnings_call_date or "",
    ) if value)
    return f"{chunk.original_text} {metadata}"


def _drop_unverifiable_quote_claims(
    response: LLMQualitativeAnalysis,
    sources: dict[str, tuple[TranscriptChunk, float]],
) -> tuple[LLMQualitativeAnalysis, int]:
    removed = 0

    def valid(item: Any) -> bool:
        nonlocal removed
        supported = all(
            bool(quote.strip()) and any(
                source_id in sources
                and _normalize(quote) in _normalize(sources[source_id][0].original_text)
                for source_id in item.source_references
            )
            for quote in item.supporting_evidence
        )
        if not supported:
            removed += 1
        return supported

    return response.model_copy(update={
        "recurring_themes": [item for item in response.recurring_themes if valid(item)],
        "management_sentiment": [item for item in response.management_sentiment if valid(item)],
    }), removed


def _unique_refs(refs: list[SourceReference]) -> list[SourceReference]:
    seen: set[str] = set()
    result: list[SourceReference] = []
    for ref in refs:
        if ref.source_id not in seen:
            seen.add(ref.source_id)
            result.append(ref)
    return result
