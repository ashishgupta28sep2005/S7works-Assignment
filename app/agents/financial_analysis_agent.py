"""Deterministic orchestration for financial extraction and qualitative RAG analysis."""

from __future__ import annotations

import json
import logging
import argparse
import re
from collections.abc import Callable, Sequence
from typing import Any

from pydantic import ValidationError

from app.config.llm import (
    GroqConfigurationError,
    GroqRequestError,
    classify_groq_exception,
    create_groq_client,
    invoke_structured,
    safe_groq_error_message,
)
from app.config.settings import get_settings
from app.prompts.master_agent_prompt import MASTER_AGENT_SYSTEM_PROMPT
from app.rag.embeddings import HuggingFaceEmbedder
from app.rag.models import TranscriptChunk
from app.rag.retriever import TranscriptRetriever
from app.rag.vector_store import PersistentFaissStore
from app.schemas.analysis import (
    CrossAnalysisFinding,
    FinancialAnalysisResult,
    FinancialMetricReference,
    LLMAnalysisResponse,
    ToolExecutionStatus,
)
from app.schemas.financial import FinancialExtractionOutput
from app.schemas.qualitative import QualitativeAnalysisResult, SourceReference
from app.tools.financial_data_extractor import FinancialDataExtractorTool
from app.tools.qualitative_analysis import (
    DEFAULT_REPORTING_PERIODS,
    QualitativeAnalysisTool,
)

logger = logging.getLogger(__name__)
_NUMBER = re.compile(r"(?<!\w)\d[\d,]*(?:\.\d+)?%?")
_TRANSIENT_GROQ_ERRORS = {"authentication", "rate_limit", "timeout", "provider_error"}


class MasterAnalysisError(RuntimeError):
    """Raised for invalid master output; its message never contains provider details."""

    def __init__(self, message: str, *, category: str = "invalid_response") -> None:
        self.category = category
        super().__init__(message)


class FinancialAnalysisAgent:
    """Run two specialist tools once, then synthesize their validated outputs once.

    Production defaults construct one shared ChatGroq client for Phase 5 and master
    reasoning. Tests inject all dependencies and never make external calls.
    """

    def __init__(
        self,
        *,
        financial_extractor: Callable[[], FinancialExtractionOutput] | Any | None = None,
        qualitative_tool: Any | None = None,
        llm: Any | None = None,
    ) -> None:
        self.llm = llm if llm is not None else create_groq_client()
        self.financial_extractor = financial_extractor or FinancialDataExtractorTool()
        self.qualitative_tool = qualitative_tool if qualitative_tool is not None else _create_qualitative_tool(self.llm)

    def analyze(self, company: str, analysis_question: str) -> FinancialAnalysisResult:
        company_name = company.strip()
        question = analysis_question.strip()
        if not company_name:
            raise ValueError("company cannot be empty")
        if not question:
            raise ValueError("analysis_question cannot be empty")
        if company_name.casefold() not in {"tcs", "tata consultancy services", "tata consultancy services limited"}:
            message = "The available reports and transcript index contain TCS documents only."
            return FinancialAnalysisResult(
                company=company_name,
                analysis_question=question,
                data_quality_notes=[message],
                tool_execution_status=ToolExecutionStatus(
                    financial_extraction="failed",
                    qualitative_analysis="failed",
                    master_reasoning="failed",
                    overall_status="failed",
                    errors={"company": message},
                ),
            )

        financial: FinancialExtractionOutput | None = None
        qualitative: QualitativeAnalysisResult | None = None
        errors: dict[str, str] = {}
        data_notes: list[str] = []

        try:
            financial = _run_financial_tool(self.financial_extractor)
            data_notes.extend(_financial_quality_notes(financial))
        except Exception as exc:
            errors["financial_extraction"] = f"Financial extraction failed ({type(exc).__name__})."
            logger.warning("Financial extraction failed (%s).", type(exc).__name__)

        qual_error_category: str | None = None
        try:
            available_periods = _available_transcript_periods(self.qualitative_tool)
            qualitative = self.qualitative_tool.analyze(question, reporting_periods=available_periods)
            if not isinstance(qualitative, QualitativeAnalysisResult):
                qualitative = QualitativeAnalysisResult.model_validate(qualitative)
            data_notes.extend(qualitative.limitations)
        except Exception as exc:
            qual_error_category = getattr(exc, "category", None) or classify_groq_exception(exc)
            errors["qualitative_analysis"] = _safe_tool_error("Qualitative analysis", qual_error_category)
            logger.warning("Qualitative analysis failed (%s).", qual_error_category)

        periods = _available_periods(financial, qualitative)
        cross_analysis: list[CrossAnalysisFinding] = []
        master_status: str = "failed"
        if financial is None and qualitative is None:
            errors["master_reasoning"] = "Master reasoning was skipped because both specialized tools failed."
        elif qual_error_category in _TRANSIENT_GROQ_ERRORS:
            errors["master_reasoning"] = (
                "Master reasoning was skipped after a Groq provider failure; a duplicate request was avoided."
            )
        else:
            try:
                response = self._reason(company_name, question, financial, qualitative)
                cross_analysis = self._resolve_cross_analysis(response, financial, qualitative)
                master_status = "success"
            except GroqRequestError as exc:
                errors["master_reasoning"] = str(exc)
                logger.warning(
                    "Master reasoning failed (%s, http_status=%s).",
                    exc.category,
                    exc.status_code if exc.status_code is not None else "unavailable",
                )
            except MasterAnalysisError as exc:
                errors["master_reasoning"] = str(exc)
                logger.warning("Master reasoning failed (%s).", exc.category)
            except Exception as exc:
                category = classify_groq_exception(exc)
                errors["master_reasoning"] = _safe_tool_error("Master reasoning", category)
                logger.warning("Master reasoning failed (%s).", category)

        specialized_count = int(financial is not None) + int(qualitative is not None)
        if specialized_count == 0:
            overall = "failed"
        elif specialized_count == 2 and master_status == "success":
            overall = "success"
        else:
            overall = "partial_success"

        status = ToolExecutionStatus(
            financial_extraction="success" if financial is not None else "failed",
            qualitative_analysis="success" if qualitative is not None else "failed",
            master_reasoning=master_status,
            overall_status=overall,
            errors=errors,
        )
        return FinancialAnalysisResult(
            company="TCS",
            analysis_question=question,
            reporting_periods=periods,
            financial_analysis=financial,
            qualitative_analysis=qualitative,
            cross_analysis=cross_analysis,
            data_quality_notes=list(dict.fromkeys(data_notes)),
            tool_execution_status=status,
        )

    def _reason(
        self,
        company: str,
        question: str,
        financial: FinancialExtractionOutput | None,
        qualitative: QualitativeAnalysisResult | None,
    ) -> LLMAnalysisResponse:
        financial_catalog = _financial_catalog(financial)
        transcript_catalog = _transcript_catalog(qualitative)
        payload = {
            "company": company,
            "analysis_question": question,
            "financial_tool_output": financial.model_dump(mode="json", exclude_none=True) if financial else None,
            "qualitative_tool_output": _qualitative_payload(qualitative) if qualitative else None,
            "financial_source_catalog": [ref.model_dump(mode="json", exclude_none=True)
                                          for ref in financial_catalog.values()],
            "transcript_source_catalog": [ref.model_dump(mode="json", exclude_none=True)
                                          for ref in transcript_catalog.values()],
        }
        messages = [
            ("system", MASTER_AGENT_SYSTEM_PROMPT),
            ("human", json.dumps(payload, ensure_ascii=False, default=str)),
        ]
        raw = invoke_structured(self.llm, LLMAnalysisResponse, messages)
        try:
            if hasattr(raw, "content"):
                raw = raw.content
            if isinstance(raw, str):
                raw = json.loads(_strip_json_fence(raw))
            return LLMAnalysisResponse.model_validate(raw)
        except (ValidationError, ValueError, TypeError):
            raise MasterAnalysisError(safe_groq_error_message("invalid_response")) from None

    @staticmethod
    def _resolve_cross_analysis(
        response: LLMAnalysisResponse,
        financial: FinancialExtractionOutput | None,
        qualitative: QualitativeAnalysisResult | None,
    ) -> list[CrossAnalysisFinding]:
        financial_catalog = _financial_catalog(financial)
        transcript_catalog = _transcript_catalog(qualitative)
        out: list[CrossAnalysisFinding] = []
        for item in response.cross_analysis:
            bad_financial = set(item.financial_source_ids) - financial_catalog.keys()
            bad_transcript = set(item.transcript_source_ids) - transcript_catalog.keys()
            if bad_financial or bad_transcript:
                raise MasterAnalysisError("Master response cited a source that was not returned by a specialist tool.")
            finding = CrossAnalysisFinding(
                finding=item.finding,
                relationship=item.relationship,
                financial_sources=[financial_catalog[source_id] for source_id in dict.fromkeys(item.financial_source_ids)],
                transcript_sources=[transcript_catalog[source_id] for source_id in dict.fromkeys(item.transcript_source_ids)],
            )
            if not _numbers_supported(finding):
                raise MasterAnalysisError("Master response contained a numerical claim unsupported by its cited sources.")
            out.append(finding)
        return out


def _create_qualitative_tool(llm: Any) -> QualitativeAnalysisTool:
    settings = get_settings()
    embedder = HuggingFaceEmbedder()
    store = PersistentFaissStore.open(
        settings.vectorstore_dir,
        dimension=embedder.dimension,
        embedding_model=settings.embedding_model,
    )
    retriever = TranscriptRetriever(store, embedder)
    return QualitativeAnalysisTool(retriever, llm=llm)


def _run_financial_tool(tool: Any) -> FinancialExtractionOutput:
    result = tool() if callable(tool) else tool.run()
    return result if isinstance(result, FinancialExtractionOutput) else FinancialExtractionOutput.model_validate(result)


def _available_transcript_periods(tool: Any) -> tuple[str, ...]:
    retriever = getattr(tool, "retriever", None)
    store = getattr(retriever, "store", None)
    chunks: Sequence[TranscriptChunk] = getattr(store, "chunks", ())
    available = tuple(sorted({chunk.reporting_period for chunk in chunks
                              if chunk.speaker_role == "management" and chunk.reporting_period}))
    return available or DEFAULT_REPORTING_PERIODS


def _available_periods(
    financial: FinancialExtractionOutput | None,
    qualitative: QualitativeAnalysisResult | None,
) -> list[str]:
    periods = {s.reporting_period for s in (financial.statements if financial else [])
               if s.reporting_period and s.reporting_period != "UNKNOWN"}
    periods.update(qualitative.reporting_periods_analyzed if qualitative else [])
    return sorted(periods)


def _financial_catalog(
    output: FinancialExtractionOutput | None,
) -> dict[str, FinancialMetricReference]:
    catalog: dict[str, FinancialMetricReference] = {}
    if output is None:
        return catalog
    for statement in output.statements:
        for name, metric in statement.financial_metrics.items():
            if metric.value is None or not metric.source_document or not metric.source_page:
                continue
            source_id = f"F{len(catalog) + 1:04d}"
            catalog[source_id] = FinancialMetricReference(
                source_id=source_id,
                metric_name=name,
                reporting_period=statement.reporting_period,
                report_period_type=statement.report_period_type,
                statement_basis=statement.statement_basis,
                value=metric.value,
                comparative_value=metric.comparative_value,
                comparative_period=metric.comparative_period,
                unit=metric.unit,
                currency=metric.currency,
                source_document=metric.source_document,
                source_page=metric.source_page,
            )
    return catalog


def _transcript_catalog(
    output: QualitativeAnalysisResult | None,
) -> dict[str, SourceReference]:
    return {reference.source_id: reference for reference in output.sources} if output else {}


def _qualitative_payload(output: QualitativeAnalysisResult) -> dict[str, Any]:
    """Keep specialist analysis, but omit score noise and never add new source claims."""
    result = output.model_dump(mode="json", exclude_none=True)
    for reference in result.get("sources", []):
        reference.pop("retrieval_score", None)
    for key in ("recurring_themes", "management_sentiment", "forward_looking_statements",
                "business_risks", "business_opportunities"):
        for item in result.get(key, []):
            for reference in item.get("source_references", []):
                reference.pop("retrieval_score", None)
    return result


def _financial_quality_notes(output: FinancialExtractionOutput) -> list[str]:
    notes = list(output.errors)
    for statement in output.statements:
        if statement.reporting_period == "UNKNOWN":
            notes.append(f"Reporting period is unidentified in {statement.source_document}.")
        missing = [name for name, metric in statement.financial_metrics.items()
                   if metric.value is None and name in {
                       "revenue", "net_profit", "total_expenses", "profit_before_tax", "eps",
                   }]
        if missing:
            notes.append(
                f"{statement.source_document} {statement.reporting_period} "
                f"({statement.statement_basis.value}) has unavailable metrics: {', '.join(missing)}."
            )
        notes.extend(f"{statement.source_document} {statement.reporting_period}: {issue}"
                     for issue in statement.validation_errors)
    return notes


def _numbers_supported(finding: CrossAnalysisFinding) -> bool:
    context = " ".join(
        f"{source.value} {source.comparative_value or ''} {source.reporting_period} "
        f"{source.comparative_period or ''} {source.unit or ''}"
        for source in finding.financial_sources
    )
    context += " " + " ".join(
        f"{source.reporting_period or ''} " for source in finding.transcript_sources
    )
    # Exact supporting quotes from specialist findings are retained in the full
    # qualitative output sent to Groq; only cited reference periods enter this
    # deterministic guard for the concise master claim.
    normalized_context = re.sub(r"[,\s]", "", context)
    return all(number.replace(",", "") in normalized_context for number in _NUMBER.findall(finding.finding))


def _safe_tool_error(tool_name: str, category: str) -> str:
    return f"{tool_name} failed: {safe_groq_error_message(category)}"


def _strip_json_fence(text: str) -> str:
    value = text.strip()
    if value.startswith("```json") and value.endswith("```"):
        return value[7:-3].strip()
    if value.startswith("```") and value.endswith("```"):
        return value[3:-3].strip()
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description="Run source-grounded TCS financial analysis.")
    parser.add_argument("--company", default="TCS", help="Company name (the current corpus is TCS-only).")
    parser.add_argument("--question", required=True, help="Analysis question for the master agent.")
    args = parser.parse_args()
    try:
        result = FinancialAnalysisAgent().analyze(args.company, args.question)
    except GroqConfigurationError as exc:
        print(f"Configuration error: {exc}")
        return 2
    except Exception as exc:
        # Deliberately avoid printing exception text from provider/client libraries.
        print(f"Analysis could not start ({type(exc).__name__}); sensitive details were withheld.")
        return 1
    print(result.model_dump_json(indent=2, exclude_none=True))
    return 0 if result.tool_execution_status.overall_status == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
