"""Mock-only tests for period-aware, evidence-grounded forecast generation."""

from __future__ import annotations

import json
import os
from decimal import Decimal

import pytest
from pydantic import ValidationError

import app.agents.forecast_agent as forecast_agent_module
from app.agents.forecast_agent import (
    ForecastAgent,
    ForecastGenerationError,
    ForecastPeriodError,
    latest_quarter,
    next_quarter,
)
from app.config.llm import create_groq_client
from app.schemas.analysis import FinancialAnalysisResult, ToolExecutionStatus
from app.schemas.financial import (
    ExtractionStatus,
    FinancialExtractionOutput,
    FinancialMetric,
    FinancialStatement,
    MetricStatus,
    StatementBasis,
)
from app.schemas.qualitative import (
    ManagementSentiment,
    QualitativeAnalysisResult,
    SourceReference,
)


def make_metric(name, value, period, document, *, page=8, unit="CRORES", basis=StatementBasis.CONSOLIDATED):
    return FinancialMetric(
        name=name, value=Decimal(str(value)) if value is not None else None,
        original_value=str(value) if value is not None else None,
        unit=unit if value is not None else None,
        currency="INR" if value is not None else None,
        reporting_period=period, statement_basis=basis,
        extraction_status=MetricStatus.EXTRACTED if value is not None else MetricStatus.NOT_FOUND,
        source_document=document if value is not None else None,
        source_page=page if value is not None else None,
        source_row_label=name.replace("_", " ").title() if value is not None else None,
    )


def make_master(*, periods=("Q1 FY27", "Q2 FY27"), missing_metrics=False,
                missing_margin=True, transcript_periods=("Q4 FY2025-26", "Q2 FY27"),
                with_conflict=False, overall="success"):
    statements = []
    for index, period in enumerate(periods):
        document = "synthetic-example-q1.pdf" if period == "Q1 FY27" else "synthetic-example-q2.pdf"
        revenue = 72275 if period == "Q1 FY27" else 73188
        profit = 13420 if period == "Q1 FY27" else 13934
        metrics = {
            "revenue": make_metric("revenue", revenue, period, document),
            "net_profit": make_metric("net_profit", None if missing_metrics else profit, period, document, page=8),
            "total_expenses": make_metric("total_expenses", 50000 + index, period, document, page=8),
            "profit_before_tax": make_metric("profit_before_tax", 17000 + index, period, document, page=8),
            "eps": make_metric("eps", 35 + index, period, document, page=8, unit="INR_PER_SHARE"),
            "operating_margin": make_metric(
                "operating_margin", None if missing_margin else 24, period, document,
                page=8, unit="PERCENT",
            ),
        }
        statements.append(FinancialStatement(
            source_document=document,
            reporting_period=period,
            report_period_type="QUARTERLY",
            statement_basis=StatementBasis.CONSOLIDATED,
            financial_metrics=metrics,
            extraction_status=ExtractionStatus.PARTIAL if missing_metrics else ExtractionStatus.SUCCESS,
        ))
    annual = FinancialStatement(
        source_document="synthetic-example-fy26.pdf", reporting_period="FY26", report_period_type="ANNUAL",
        statement_basis=StatementBasis.CONSOLIDATED,
        financial_metrics={"revenue": make_metric("revenue", 250000, "FY26", "synthetic-example-fy26.pdf")},
        extraction_status=ExtractionStatus.SUCCESS,
    )
    statements.append(annual)

    transcript_refs = []
    sentiments = []
    for index, period in enumerate(transcript_periods):
        ref = SourceReference(
            source_id=f"S{index + 1:03d}",
            document_name="synthetic-example-call-q2.pdf" if period == "Q2 FY27" else "synthetic-example-call-q4.pdf",
            reporting_period=period, page_number=5 if period == "Q2 FY27" else 29,
            speaker="Samir Seksaria" if period == "Q2 FY27" else "K Krithivasan",
            retrieval_score=0.8,
        )
        transcript_refs.append(ref)
        quote = (
            "In constant currency terms, our growth was 50 basis points QoQ and 2.8% YoY."
            if period == "Q2 FY27" else "Our margins are very strong."
        )
        sentiments.append(ManagementSentiment(
            sentiment="neutral",
            explanation="Management described recent business conditions.",
            supporting_evidence=[quote],
            source_references=[ref],
        ))
    qualitative = QualitativeAnalysisResult(
        company="Example Corp", analysis_question="Forecast outlook",
        reporting_periods_analyzed=list(transcript_periods),
        management_sentiment=sentiments,
        sources=transcript_refs,
    )
    result = FinancialAnalysisResult(
        company="Example Corp", analysis_question="Forecast outlook",
        reporting_periods=[*periods, "FY26", *transcript_periods],
        financial_analysis=FinancialExtractionOutput(statements=statements),
        qualitative_analysis=qualitative,
        data_quality_notes=["Synthetic Example Corp fixture for technical tests only."],
        tool_execution_status=ToolExecutionStatus(
            financial_extraction="success", qualitative_analysis="success",
            master_reasoning="success", overall_status=overall,
        ),
    )
    if with_conflict:
        from app.schemas.analysis import CrossAnalysisFinding, FinancialMetricReference
        result.cross_analysis = [CrossAnalysisFinding(
            finding="Management commentary and financial evidence diverge.",
            relationship="divergence",
            financial_sources=[FinancialMetricReference(
                source_id="F0001", metric_name="revenue", reporting_period=periods[-1],
                report_period_type="QUARTERLY", statement_basis=StatementBasis.CONSOLIDATED,
                value=Decimal("73188"), unit="CRORES", currency="INR",
                source_document="synthetic-example-q2.pdf", source_page=8,
            )],
            transcript_sources=[transcript_refs[-1]],
        )]
    return result


def llm_response(*, references=("F0001", "F0003", "S002"), confidence="high", conflict=False):
    refs = list(references)
    transcript_ref = next((ref for ref in reversed(refs) if ref.startswith("S")), "S001")
    margin = {
        "direction": "uncertain" if not conflict else "cautious",
        "evidence_status": "unavailable" if not conflict else "conflicting",
        "outlook": "Operating margin is unavailable in the extracted financial statements."
        if not conflict else "Conflicting evidence prevents a directional margin conclusion.",
        "source_references": [] if not conflict else [refs[-1]],
    }
    return {
        "overall_business_outlook": {
            "direction": "cautious", "evidence_status": "limited",
            "outlook": "Recent evidence supports cautious optimism, but it does not establish that the trend will continue.",
            "source_references": refs,
        },
        "revenue_outlook": {
            "direction": "stable", "evidence_status": "supported",
            "outlook": "Recent consolidated quarterly revenue increased, while management described growth as a priority.",
            "source_references": refs,
        },
        "profitability_outlook": {
            "direction": "cautious", "evidence_status": "limited",
            "outlook": "Recent reported profit improved, but the short history limits confidence in continuation.",
        "source_references": refs,
        },
        "operating_margin_outlook": margin,
        "management_outlook": {
            "direction": "stable", "evidence_status": "limited",
            "outlook": "Management described historical growth and investment priorities; these comments are not a commitment.",
        "source_references": [transcript_ref],
        },
        "business_risks": [],
        "business_opportunities": [],
        "key_growth_drivers": [{
            "item": "Management's stated growth prioritization",
            "reasoning": "The transcript records management describing growth as a business priority.",
            "source_references": [transcript_ref],
        }],
        "key_monitoring_indicators": [{
            "item": "Quarterly consolidated revenue and net profit",
            "reasoning": "These extracted metrics provide observable follow-up evidence.",
        "source_references": refs,
        }],
        "confidence_level": confidence,
        "reasoning": [{
            "inference": "The recent quarter and management commentary provide a cautious rather than assured outlook.",
            "source_references": refs,
        }],
    }


class FakeMaster:
    def __init__(self, result=None):
        self.result = result if result is not None else make_master()
        self.calls = []

    def analyze(self, company, question):
        self.calls.append((company, question))
        return self.result


class FakeLLM:
    def __init__(self, response=None, error=None):
        self.response = response if response is not None else llm_response()
        self.error = error
        self.calls = 0
        self.messages = None
        self.schema = None

    def with_structured_output(self, schema, **kwargs):
        self.schema = schema
        self.structured_kwargs = kwargs
        return self

    def invoke(self, messages):
        self.calls += 1
        self.messages = messages
        if self.error:
            raise self.error
        return self.response


def run(response=None, *, result=None, error=None):
    master = FakeMaster(result)
    llm = FakeLLM(response, error)
    forecast = ForecastAgent(master_agent=master, llm=llm).generate(
        "Example Corp", "Generate a qualitative outlook for the upcoming quarter.",
    )
    return forecast, master, llm


def test_successful_forecast_reuses_master_once_and_preserves_provenance():
    result, master, llm = run()
    assert master.calls == [("Example Corp", "Generate a qualitative outlook for the upcoming quarter.")]
    assert llm.calls == 1
    assert llm.schema.__name__ == "LLMQuarterlyForecast"
    assert llm.structured_kwargs == {"method": "json_schema", "strict": False}
    assert result.forecast_for == "Q3 FY27"
    assert result.based_on_reporting_periods == ["Q4 FY2025-26", "Q1 FY27", "Q2 FY27"]
    assert result.confidence_level == "medium"  # Limited to two transcript periods.
    assert result.revenue_outlook.source_references[0].source.source_document == "synthetic-example-q1.pdf"
    assert result.management_outlook.source_references[0].source.speaker == "Samir Seksaria"


def test_period_detection_handles_quarter_order_and_fiscal_year_rollover():
    master = make_master(periods=("Q1 FY27", "Q2 FY27"))
    master.reporting_periods = ["FY26", "Q4 FY2025-26", "Q1 FY27", "Q2 FY27"]
    assert latest_quarter(master) == "Q2 FY27"
    assert next_quarter("Q2 FY27") == "Q3 FY27"
    assert next_quarter("Q4 FY2025-26") == "Q1 FY27"


def test_missing_reporting_period_raises_without_guessing():
    master = make_master(periods=(), transcript_periods=())
    master.reporting_periods = ["FY26"]
    with pytest.raises(ForecastPeriodError, match="no reliably identified quarterly period"):
        latest_quarter(master)
    fake_master = FakeMaster(master)
    llm = FakeLLM()
    with pytest.raises(ForecastPeriodError):
        ForecastAgent(fake_master, llm).generate("TCS", "Outlook")
    assert llm.calls == 0


def test_missing_operating_margin_is_reported_as_unavailable_not_inferred():
    result, _, _ = run()
    assert result.operating_margin_outlook.evidence_status == "unavailable"
    assert result.operating_margin_outlook.direction == "uncertain"
    assert not result.operating_margin_outlook.source_references
    assert "not available" in result.operating_margin_outlook.outlook
    assert any("Operating margin is unavailable" in note for note in result.data_limitations)


def test_insufficient_financial_evidence_lowers_confidence_and_keeps_nulls():
    master = make_master(missing_metrics=True, transcript_periods=("Q2 FY27",))
    response = llm_response(references=("F0001", "S001"))
    result, _, _ = run(response, result=master)
    assert result.confidence_level == "low"
    assert any("unavailable metrics" in note for note in result.data_limitations)
    assert master.financial_analysis.statements[1].financial_metrics["net_profit"].value is None


def test_conflicting_phase6_evidence_is_included_as_a_limitation():
    master = make_master(with_conflict=True)
    response = llm_response(conflict=True)
    result, _, llm = run(response, result=master)
    payload = json.loads(llm.messages[1][1])
    assert payload["phase6_cross_analysis"][0]["relationship"] == "divergence"
    assert result.operating_margin_outlook.evidence_status == "unavailable"  # Missing extracted metric wins.
    assert any("conflicting quantitative and management evidence" in note for note in result.data_limitations)


def test_unsupported_numerical_forecast_is_rejected():
    response = llm_response()
    response["revenue_outlook"]["outlook"] = "Revenue will grow 10% next quarter."
    with pytest.raises(ForecastGenerationError, match="numerical figure") as error:
        run(response)
    assert error.value.category == "unsupported_numeric_forecast"


def test_invalid_source_reference_is_rejected():
    response = llm_response()
    response["revenue_outlook"]["source_references"] = ["F9999"]
    with pytest.raises(ForecastGenerationError, match="not present in the Phase 6 outputs") as error:
        run(response)
    assert error.value.category == "invalid_source_reference"


@pytest.mark.parametrize(("exception_type", "category", "message"), [
    ("AuthenticationError", "authentication", "authentication failed"),
    ("RateLimitError", "rate_limit", "rate limit reached"),
    ("APITimeoutError", "timeout", "timed out"),
])
def test_groq_errors_are_categorized_without_sensitive_details(exception_type, category, message):
    exc = type(exception_type, (Exception,), {})()
    with pytest.raises(ForecastGenerationError) as error:
        run(error=exc)
    assert error.value.category == category
    assert message in str(error.value).lower()


def test_invalid_json_and_pydantic_validation_errors_are_safe():
    with pytest.raises(ForecastGenerationError) as json_error:
        run("not json")
    assert json_error.value.category == "invalid_response"

    response = llm_response()
    response["confidence_level"] = "certain"
    with pytest.raises(ForecastGenerationError) as validation_error:
        run(response)
    assert validation_error.value.category == "invalid_response"
    assert "certain" not in str(validation_error.value)


def test_low_confidence_is_preserved_even_with_only_one_transcript_period():
    master = make_master(transcript_periods=("Q2 FY27",))
    response = llm_response(confidence="low", references=("F0001", "S001"))
    result, _, _ = run(response, result=master)
    assert result.confidence_level == "low"
    assert result.data_limitations


def test_annual_and_standalone_evidence_are_not_sent_as_forecast_metrics():
    result, _, llm = run()
    payload = json.loads(llm.messages[1][1])
    evidence = payload["quarterly_consolidated_financial_evidence"]
    assert evidence
    assert all(row["report_period_type"] == "QUARTERLY" for row in evidence)
    assert all(row["statement_basis"] == "CONSOLIDATED" for row in evidence)
    assert all("FY26" not in row["reporting_period"] for row in evidence)
    assert result.forecast_for == "Q3 FY27"


def test_no_evidence_returns_uncertain_forecast_without_llm_call():
    master = make_master(periods=(), transcript_periods=())
    master.reporting_periods = ["Q2 FY27"]
    master.financial_analysis = FinancialExtractionOutput(statements=[])
    master.qualitative_analysis = QualitativeAnalysisResult(
        company="TCS", analysis_question="outlook", reporting_periods_analyzed=[],
    )
    agent = ForecastAgent(FakeMaster(master), FakeLLM())
    # The explicitly identified master period provides the horizon, but no source supports claims.
    result = agent.generate("TCS", "outlook")
    assert result.forecast_for == "Q3 FY27"
    assert result.confidence_level == "low"
    assert result.overall_business_outlook.evidence_status == "unavailable"


def test_module_entry_point_prints_json(monkeypatch, capsys):
    monkeypatch.setattr(
        forecast_agent_module,
        "ForecastAgent",
        lambda: ForecastAgent(master_agent=FakeMaster(make_master()), llm=FakeLLM()),
    )
    assert forecast_agent_module.main(["--company", "Example Corp", "--question", "Outlook"]) == 0
    assert json.loads(capsys.readouterr().out)["forecast_for"] == "Q3 FY27"


@pytest.mark.skipif(
    os.getenv("RUN_GROQ_INTEGRATION_TESTS", "false").casefold() != "true",
    reason="Set RUN_GROQ_INTEGRATION_TESTS=true to make the optional live Groq call.",
)
def test_optional_live_groq_forecast_integration():
    # Technical integration fixture only: fictional Example Corp data, not TCS evidence.
    forecast = ForecastAgent(
        master_agent=FakeMaster(make_master()),
        llm=create_groq_client(),
    ).generate("Example Corp", "Return a cautious qualitative outlook for the next quarter.")
    assert forecast.forecast_for == "Q3 FY27"
    assert forecast.company == "Example Corp"
    assert forecast.source_references
