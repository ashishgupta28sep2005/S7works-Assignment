"""Generate an evidence-grounded qualitative outlook for the next TCS quarter."""

from __future__ import annotations

import json
import logging
import re
import argparse
from collections import defaultdict
from typing import Any

from pydantic import ValidationError

from app.agents.financial_analysis_agent import FinancialAnalysisAgent
from app.config.llm import (
    GroqConfigurationError,
    GroqRequestError,
    create_groq_client,
    invoke_structured,
    safe_groq_error_message,
)
from app.prompts.forecast_prompt import FORECAST_SYSTEM_PROMPT
from app.schemas.analysis import FinancialAnalysisResult, FinancialMetricReference
from app.schemas.financial import FinancialMetric, FinancialStatement, StatementBasis
from app.schemas.forecast import (
    ForecastAssessment,
    ForecastItem,
    ForecastReasoning,
    ForecastSourceReference,
    LLMQuarterlyForecast,
    QuarterlyForecast,
)
from app.schemas.qualitative import SourceReference

logger = logging.getLogger(__name__)
_QUARTER = re.compile(r"^Q([1-4])\s+FY\s*(\d{2,4})(?:\s*-\s*(\d{2,4}))?$", re.I)
_ANY_NUMBER = re.compile(r"(?<!\w)\d[\d,]*(?:\.\d+)?%?")


class ForecastGenerationError(RuntimeError):
    """Safe forecast failure, with a stable non-sensitive category."""

    def __init__(self, message: str, *, category: str = "invalid_response") -> None:
        self.category = category
        super().__init__(message)


class ForecastPeriodError(ValueError):
    """Raised when the next quarter cannot be derived from explicit source periods."""


class ForecastAgent:
    """Use Phase 6 once, then make a single structured forecast synthesis request.

    By default the Phase 6 agent and this agent share one Groq client. Tests inject
    a mock master agent and mock LLM; ordinary test runs never contact Groq.
    """

    def __init__(self, master_agent: Any | None = None, llm: Any | None = None) -> None:
        self.master_agent = master_agent if master_agent is not None else FinancialAnalysisAgent()
        shared_llm = getattr(self.master_agent, "llm", None)
        self.llm = llm if llm is not None else shared_llm if shared_llm is not None else create_groq_client()

    def generate(self, company: str, forecast_request: str) -> QuarterlyForecast:
        company_name = company.strip()
        request = forecast_request.strip()
        if not company_name:
            raise ValueError("company cannot be empty")
        if not request:
            raise ValueError("forecast_request cannot be empty")

        master = self.master_agent.analyze(company_name, request)
        if not isinstance(master, FinancialAnalysisResult):
            try:
                master = FinancialAnalysisResult.model_validate(master)
            except ValidationError:
                raise ForecastGenerationError(
                    "Phase 6 returned an invalid consolidated analysis.", category="invalid_response",
                ) from None
        if master.tool_execution_status.overall_status == "failed":
            provider_category = _phase6_provider_error_category(master)
            raise ForecastGenerationError(
                "Forecast generation requires at least one successful Phase 6 specialist result."
                + (f" {next(iter(master.tool_execution_status.errors.values()))}" if provider_category else ""),
                category=provider_category or "insufficient_evidence",
            )

        latest = latest_quarter(master)
        forecast_for = next_quarter(latest)
        based_periods = _based_on_periods(master)
        if not based_periods:
            # Use the explicit Phase 6 period even when specialist outputs are empty.
            based_periods = [latest]
        source_map, transcript_evidence = _source_catalog(master)
        limitations = _base_limitations(master, source_map)
        if not source_map:
            return _empty_forecast(
                master, company_name, forecast_for, based_periods,
                limitations + ["No source references were available for a grounded forecast."],
            )

        financial_rows = _quarterly_consolidated_evidence(master, source_map)
        transcript_rows = _transcript_evidence(master, source_map, transcript_evidence)
        payload = {
            "company": master.company,
            "forecast_request": request,
            "forecast_for": forecast_for,
            "based_on_reporting_periods": based_periods,
            "master_overall_status": master.tool_execution_status.overall_status,
            "master_data_quality_notes": master.data_quality_notes,
            "quarterly_consolidated_financial_evidence": financial_rows,
            "management_transcript_evidence": transcript_rows,
            "phase6_cross_analysis": [
                finding.model_dump(mode="json", exclude_none=True)
                for finding in master.cross_analysis
                if not finding.financial_sources or all(
                    source.report_period_type == "QUARTERLY"
                    and source.statement_basis == StatementBasis.CONSOLIDATED
                    for source in finding.financial_sources
                )
            ],
            "operating_margin_available_in_latest_consolidated_quarter": _operating_margin_available(
                master, latest,
            ),
            "transcript_reporting_period_count": len(
                master.qualitative_analysis.reporting_periods_analyzed
                if master.qualitative_analysis else []
            ),
        }
        try:
            raw = invoke_structured(self.llm, LLMQuarterlyForecast, [
                ("system", FORECAST_SYSTEM_PROMPT),
                ("human", json.dumps(payload, ensure_ascii=False, default=str)),
            ])
        except GroqRequestError as exc:
            raise ForecastGenerationError(str(exc), category=exc.category) from None

        try:
            if hasattr(raw, "content"):
                raw = raw.content
            if isinstance(raw, str):
                raw = json.loads(_strip_json_fence(raw))
            response = LLMQuarterlyForecast.model_validate(raw)
        except (ValidationError, ValueError, TypeError):
            raise ForecastGenerationError(
                safe_groq_error_message("invalid_response"), category="invalid_response",
            ) from None

        _validate_forecast_prose(response)
        _validate_source_ids(response, source_map)
        forecast = _resolve_response(
            response, master, company_name, forecast_for, based_periods,
            source_map, limitations,
        )
        return forecast


def latest_quarter(master: FinancialAnalysisResult) -> str:
    """Find the latest explicit quarterly reporting period, never infer from dates."""
    candidates: set[str] = set()
    financial = master.financial_analysis
    if financial:
        candidates.update(
            statement.reporting_period
            for statement in financial.statements
            if statement.report_period_type == "QUARTERLY"
            and statement.reporting_period != "UNKNOWN"
            and _parse_quarter(statement.reporting_period) is not None
        )
    if master.qualitative_analysis:
        candidates.update(
            period for period in master.qualitative_analysis.reporting_periods_analyzed
            if _parse_quarter(period) is not None
        )
    # Master periods are explicitly sourced from those same Phase 6 outputs.
    candidates.update(period for period in master.reporting_periods if _parse_quarter(period) is not None)
    if not candidates:
        raise ForecastPeriodError(
            "Cannot determine the upcoming quarter: Phase 6 contains no reliably identified quarterly period."
        )
    return max(candidates, key=lambda period: _parse_quarter(period) or (0, 0))


def next_quarter(reporting_period: str) -> str:
    parsed = _parse_quarter(reporting_period)
    if parsed is None:
        raise ForecastPeriodError(f"Cannot determine the next quarter from {reporting_period!r}.")
    fiscal_year, quarter = parsed
    if quarter == 4:
        quarter = 1
        fiscal_year = fiscal_year % 100 + 1
    else:
        quarter += 1
    return f"Q{quarter} FY{fiscal_year % 100:02d}"


def _parse_quarter(period: str) -> tuple[int, int] | None:
    match = _QUARTER.fullmatch(period.strip())
    if not match:
        return None
    quarter = int(match.group(1))
    fiscal_year = int(match.group(3) or match.group(2))
    if fiscal_year >= 100:
        fiscal_year %= 100
    return fiscal_year, quarter


def _based_on_periods(master: FinancialAnalysisResult) -> list[str]:
    periods: set[str] = set()
    if master.financial_analysis:
        periods.update(
            statement.reporting_period
            for statement in master.financial_analysis.statements
            if statement.report_period_type == "QUARTERLY"
            and statement.statement_basis == StatementBasis.CONSOLIDATED
            and _parse_quarter(statement.reporting_period) is not None
        )
    if master.qualitative_analysis:
        periods.update(
            period for period in master.qualitative_analysis.reporting_periods_analyzed
            if _parse_quarter(period) is not None
        )
    return sorted(periods, key=lambda value: _parse_quarter(value) or (0, 0))


def _source_catalog(
    master: FinancialAnalysisResult,
) -> tuple[dict[str, ForecastSourceReference], dict[str, list[str]]]:
    catalog: dict[str, ForecastSourceReference] = {}
    evidence: dict[str, list[str]] = defaultdict(list)
    financial = master.financial_analysis
    if financial:
        for statement in financial.statements:
            if (statement.report_period_type != "QUARTERLY"
                    or statement.statement_basis != StatementBasis.CONSOLIDATED):
                continue
            for metric_name, metric in statement.financial_metrics.items():
                if metric.value is None or not metric.source_document or not metric.source_page:
                    continue
                if _parse_quarter(statement.reporting_period) is None:
                    continue
                source_id = f"F{len([key for key in catalog if key.startswith('F')]) + 1:04d}"
                reference = FinancialMetricReference(
                    source_id=source_id,
                    metric_name=metric_name,
                    reporting_period=statement.reporting_period,
                    report_period_type="QUARTERLY",
                    statement_basis=StatementBasis.CONSOLIDATED,
                    value=metric.value,
                    comparative_value=metric.comparative_value,
                    comparative_period=metric.comparative_period,
                    unit=metric.unit,
                    currency=metric.currency,
                    source_document=metric.source_document,
                    source_page=metric.source_page,
                )
                # Future forecasts should use only matching-quarter comparatives.
                if reference.comparative_period and _parse_quarter(reference.comparative_period) is None:
                    reference = reference.model_copy(update={
                        "comparative_value": None,
                        "comparative_period": None,
                    })
                catalog[source_id] = ForecastSourceReference(
                    source=reference,
                    supporting_evidence=[
                        f"{metric.source_row_label}: {metric.original_value}"
                        if metric.source_row_label and metric.original_value
                        else f"{metric_name}: {metric.value} {metric.unit or ''}".strip()
                    ],
                )
    qualitative = master.qualitative_analysis
    if qualitative:
        for reference in qualitative.sources:
            catalog[reference.source_id] = ForecastSourceReference(source=reference)
        for category in (
            qualitative.recurring_themes,
            qualitative.management_sentiment,
        ):
            for item in category:
                for reference in item.source_references:
                    evidence[reference.source_id].extend(item.supporting_evidence)
        # Other Phase 6 claims have no quote field; their cited, validated source
        # metadata remains available, but no quotation is added or fabricated.
    for source_id, quotes in evidence.items():
        reference = catalog.get(source_id)
        if reference is not None:
            catalog[source_id] = reference.model_copy(update={
                "supporting_evidence": list(dict.fromkeys(quotes)),
            })
    return catalog, evidence


def _quarterly_consolidated_evidence(
    master: FinancialAnalysisResult,
    catalog: dict[str, ForecastSourceReference],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for source_id, wrapper in catalog.items():
        ref = wrapper.source
        if not isinstance(ref, FinancialMetricReference):
            continue
        statement: FinancialStatement | None = None
        if master.financial_analysis:
            statement = next((item for item in master.financial_analysis.statements
                              if item.reporting_period == ref.reporting_period
                              and item.statement_basis == ref.statement_basis
                              and item.report_period_type == ref.report_period_type), None)
        metric: FinancialMetric | None = statement.financial_metrics.get(ref.metric_name) if statement else None
        output.append({
            "source_id": source_id,
            **ref.model_dump(mode="json", exclude_none=True),
            "original_value": metric.original_value if metric else None,
            "source_row_label": metric.source_row_label if metric else None,
            "comparative_original_value": metric.comparative_original_value if metric else None,
            "extraction_status": metric.extraction_status.value if metric else None,
        })
    return output


def _transcript_evidence(
    master: FinancialAnalysisResult,
    catalog: dict[str, ForecastSourceReference],
    evidence: dict[str, list[str]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for source_id, wrapper in catalog.items():
        if not isinstance(wrapper.source, SourceReference):
            continue
        output.append({
            "source_id": source_id,
            **wrapper.source.model_dump(mode="json", exclude_none=True),
            "supporting_evidence": list(dict.fromkeys(evidence.get(source_id, []))),
        })
    return output


def _operating_margin_available(master: FinancialAnalysisResult, latest: str) -> bool:
    if not master.financial_analysis:
        return False
    return any(
        statement.reporting_period == latest
        and statement.report_period_type == "QUARTERLY"
        and statement.statement_basis == StatementBasis.CONSOLIDATED
        and (metric := statement.financial_metrics.get("operating_margin")) is not None
        and metric.value is not None
        and metric.source_document is not None
        and metric.source_page is not None
        for statement in master.financial_analysis.statements
    )


def _validate_forecast_prose(response: LLMQuarterlyForecast) -> None:
    text: list[str] = [
        response.overall_business_outlook.outlook,
        response.revenue_outlook.outlook,
        response.profitability_outlook.outlook,
        response.operating_margin_outlook.outlook,
        response.management_outlook.outlook,
    ]
    text.extend(f"{item.item} {item.reasoning}" for item in (
        *response.business_risks,
        *response.business_opportunities,
        *response.key_growth_drivers,
        *response.key_monitoring_indicators,
    ))
    text.extend(item.inference for item in response.reasoning)
    if any(_ANY_NUMBER.search(value) for value in text):
        raise ForecastGenerationError(
            "Forecast prose contained a numerical figure; numerical forecasts are not permitted.",
            category="unsupported_numeric_forecast",
        )


def _validate_source_ids(
    response: LLMQuarterlyForecast,
    catalog: dict[str, ForecastSourceReference],
) -> None:
    all_ids: set[str] = set()
    for assessment in _assessments(response):
        all_ids.update(assessment.source_references)
    for item in _items(response):
        all_ids.update(item.source_references)
    for item in response.reasoning:
        all_ids.update(item.source_references)
    invalid = all_ids - catalog.keys()
    if invalid:
        raise ForecastGenerationError(
            "Forecast cited source IDs that were not present in the Phase 6 outputs.",
            category="invalid_source_reference",
        )


def _resolve_response(
    response: LLMQuarterlyForecast,
    master: FinancialAnalysisResult,
    company: str,
    forecast_for: str,
    based_periods: list[str],
    catalog: dict[str, ForecastSourceReference],
    limitations: list[str],
) -> QuarterlyForecast:
    resolved = {
        name: _resolve_assessment(getattr(response, name), catalog)
        for name in (
            "overall_business_outlook", "revenue_outlook", "profitability_outlook",
            "operating_margin_outlook", "management_outlook",
        )
    }
    if not _operating_margin_available(master, latest_quarter(master)):
        resolved["operating_margin_outlook"] = ForecastAssessment(
            direction="uncertain",
            evidence_status="unavailable",
            outlook=(
                "Operating margin was not available as a source-linked metric in the latest "
                "consolidated quarterly financial extraction; no operating-margin outlook is provided."
            ),
            source_references=[],
        )
        limitations.append(
            "Operating margin is unavailable in the latest source-linked consolidated quarterly financial extraction."
        )

    risks = [_resolve_item(item, catalog) for item in response.business_risks]
    opportunities = [_resolve_item(item, catalog) for item in response.business_opportunities]
    drivers = [_resolve_item(item, catalog) for item in response.key_growth_drivers]
    indicators = [_resolve_item(item, catalog) for item in response.key_monitoring_indicators]
    reasoning = [ForecastReasoning(
        inference=item.inference,
        source_references=_resolve_ids(item.source_references, catalog),
    ) for item in response.reasoning]
    all_refs = _unique_references(
        [ref for assessment in resolved.values() for ref in assessment.source_references]
        + [ref for item in (*risks, *opportunities, *drivers, *indicators) for ref in item.source_references]
        + [ref for item in reasoning for ref in item.source_references]
    )

    # A two-period transcript history is necessarily limited for forecasting.
    confidence = response.confidence_level
    transcript_period_count = len(
        master.qualitative_analysis.reporting_periods_analyzed if master.qualitative_analysis else []
    )
    core_metrics_incomplete = _core_metrics_incomplete(master, latest_quarter(master))
    if core_metrics_incomplete:
        limitations.append(
            "One or more unavailable metrics in the latest quarterly consolidated period reduce confidence."
        )
    if transcript_period_count < 2 or core_metrics_incomplete or master.tool_execution_status.overall_status != "success":
        confidence = "low"
    elif transcript_period_count == 2 and confidence == "high":
        confidence = "medium"

    if transcript_period_count == 2:
        limitations.append(
            "Only two earnings-call reporting periods are represented in the available transcript evidence."
        )
    if any(finding.relationship == "divergence" for finding in master.cross_analysis):
        limitations.append(
            "Phase 6 identified conflicting quantitative and management evidence; the outlook preserves this uncertainty."
        )
    limitations.extend(master.data_quality_notes)
    return QuarterlyForecast(
        company=master.company or company,
        forecast_for=forecast_for,
        based_on_reporting_periods=based_periods,
        **resolved,
        business_risks=risks,
        business_opportunities=opportunities,
        key_growth_drivers=drivers,
        key_monitoring_indicators=indicators,
        confidence_level=confidence,
        reasoning=reasoning,
        source_references=all_refs,
        data_limitations=list(dict.fromkeys(limitations)),
    )


def _resolve_assessment(
    item: Any,
    catalog: dict[str, ForecastSourceReference],
) -> ForecastAssessment:
    return ForecastAssessment(
        direction=item.direction,
        evidence_status=item.evidence_status,
        outlook=item.outlook,
        source_references=_resolve_ids(item.source_references, catalog),
    )


def _resolve_item(item: Any, catalog: dict[str, ForecastSourceReference]) -> ForecastItem:
    return ForecastItem(
        item=item.item,
        reasoning=item.reasoning,
        source_references=_resolve_ids(item.source_references, catalog),
    )


def _resolve_ids(
    ids: list[str], catalog: dict[str, ForecastSourceReference],
) -> list[ForecastSourceReference]:
    return [catalog[source_id] for source_id in dict.fromkeys(ids)]


def _assessments(response: LLMQuarterlyForecast) -> list[Any]:
    return [getattr(response, name) for name in (
        "overall_business_outlook", "revenue_outlook", "profitability_outlook",
        "operating_margin_outlook", "management_outlook",
    )]


def _items(response: LLMQuarterlyForecast) -> list[Any]:
    return [item for category in (
        response.business_risks, response.business_opportunities,
        response.key_growth_drivers, response.key_monitoring_indicators,
    ) for item in category]


def _unique_references(refs: list[ForecastSourceReference]) -> list[ForecastSourceReference]:
    unique: dict[str, ForecastSourceReference] = {}
    for ref in refs:
        unique.setdefault(ref.source.source_id, ref)
    return list(unique.values())


def _core_metrics_incomplete(master: FinancialAnalysisResult, latest: str) -> bool:
    financial = master.financial_analysis
    if not financial:
        return True
    statements = [statement for statement in financial.statements
                  if statement.reporting_period == latest
                  and statement.report_period_type == "QUARTERLY"
                  and statement.statement_basis == StatementBasis.CONSOLIDATED]
    if not statements:
        return True
    statement = statements[0]
    return any(statement.financial_metrics.get(name) is None
               or statement.financial_metrics[name].value is None
               for name in ("revenue", "net_profit"))


def _base_limitations(master: FinancialAnalysisResult, catalog: dict[str, ForecastSourceReference]) -> list[str]:
    notes = list(master.data_quality_notes)
    if master.tool_execution_status.overall_status != "success":
        notes.append("Phase 6 completed with partial or failed specialist evidence; forecast confidence is reduced.")
    if not any(isinstance(ref.source, FinancialMetricReference) for ref in catalog.values()):
        notes.append("No source-linked quarterly consolidated financial metrics were available for the forecast.")
    if not any(isinstance(ref.source, SourceReference) for ref in catalog.values()):
        notes.append("No source-linked management transcript evidence was available for the forecast.")
    return notes


def _empty_forecast(
    master: FinancialAnalysisResult,
    company: str,
    forecast_for: str,
    based_periods: list[str],
    limitations: list[str],
) -> QuarterlyForecast:
    unavailable = ForecastAssessment(
        direction="uncertain",
        evidence_status="unavailable",
        outlook="Insufficient source evidence is available to form this outlook.",
        source_references=[],
    )
    if not limitations:
        limitations = ["No evidence is available to support a qualitative outlook."]
    return QuarterlyForecast(
        company=master.company or company,
        forecast_for=forecast_for,
        based_on_reporting_periods=based_periods,
        overall_business_outlook=unavailable,
        revenue_outlook=unavailable,
        profitability_outlook=unavailable,
        operating_margin_outlook=unavailable.model_copy(update={
            "outlook": "Operating margin is unavailable in the extracted evidence; no margin outlook is provided."
        }),
        management_outlook=unavailable,
        confidence_level="low",
        data_limitations=limitations,
    )


def _strip_json_fence(text: str) -> str:
    value = text.strip()
    if value.startswith("```json") and value.endswith("```"):
        return value[7:-3].strip()
    if value.startswith("```") and value.endswith("```"):
        return value[3:-3].strip()
    return value


def _phase6_provider_error_category(master: FinancialAnalysisResult) -> str | None:
    messages = " ".join(master.tool_execution_status.errors.values()).casefold()
    for token, category in (
        ("authentication failed", "authentication"),
        ("rate limit", "rate_limit"),
        ("timed out", "timeout"),
        ("structured schema", "invalid_response"),
    ):
        if token in messages:
            return category
    return None


def main(argv: list[str] | None = None) -> int:
    """Run a single upcoming-quarter forecast from the project root."""
    parser = argparse.ArgumentParser(description="Generate a source-grounded qualitative TCS forecast.")
    parser.add_argument("--company", default="TCS", help="Company name (the current corpus is TCS-only).")
    parser.add_argument("--question", required=True, help="Forecast request for the upcoming quarter.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        result = ForecastAgent().generate(args.company, args.question)
    except GroqConfigurationError as exc:
        print(f"Configuration error: {exc}")
        return 2
    except ForecastGenerationError as exc:
        print(f"Forecast failed [{exc.category}]: {exc}")
        return 1
    except Exception as exc:
        # Avoid emitting arbitrary provider/client exception text which may contain request details.
        logger.exception("Forecast failed unexpectedly (%s).", type(exc).__name__)
        print(f"Forecast failed ({type(exc).__name__}); sensitive details were withheld.")
        return 1
    print(result.model_dump_json(indent=2, exclude_none=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
