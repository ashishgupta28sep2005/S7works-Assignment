"""Deterministic orchestration tests with mocked tools and Groq responses."""

from __future__ import annotations

import json
from decimal import Decimal

import pytest
from pydantic import ValidationError

from app.schemas.analysis import FinancialAnalysisResult
from app.config.settings import Settings
from app.config.llm import classify_groq_exception, create_groq_client, invoke_structured
from app.schemas.financial import (
    ExtractionStatus, FinancialExtractionOutput, FinancialMetric, FinancialStatement,
    MetricStatus, StatementBasis,
)
from app.schemas.qualitative import (
    ManagementSentiment, QualitativeAnalysisResult, SourceReference,
)
from app.agents import financial_analysis_agent as agent_module
from app.agents.financial_analysis_agent import FinancialAnalysisAgent
from app.tools.qualitative_analysis import QualitativeAnalysisError


def financial_result(*, incomplete=False, missing_period=False):
    reports = []
    for period, period_type, basis, revenue in (
        ("Q2 FY27", "QUARTERLY", StatementBasis.CONSOLIDATED, Decimal("65454.0")),
        ("Q2 FY27", "QUARTERLY", StatementBasis.STANDALONE, Decimal("61000.0")),
        ("FY26", "ANNUAL", StatementBasis.CONSOLIDATED, Decimal("250000.0")),
    ):
        period_value = "UNKNOWN" if missing_period and period_type == "ANNUAL" else period
        metrics = {
            "revenue": FinancialMetric(
                name="revenue", value=revenue, reporting_period=period_value,
                statement_basis=basis, extraction_status=MetricStatus.EXTRACTED,
                source_document=f"report-{period}-{basis.value}.pdf", source_page=5,
                unit="CRORES", currency="INR",
            ),
            "net_profit": FinancialMetric(
                name="net_profit", value=None if incomplete else Decimal("12000"),
                reporting_period=period_value, statement_basis=basis,
                extraction_status=MetricStatus.NOT_FOUND if incomplete else MetricStatus.EXTRACTED,
                source_document=None if incomplete else f"report-{period}-{basis.value}.pdf",
                source_page=None if incomplete else 6,
            ),
        }
        reports.append(FinancialStatement(
            source_document=f"report-{period}-{basis.value}.pdf", reporting_period=period_value,
            report_period_type=period_type, statement_basis=basis,
            financial_metrics=metrics,
            extraction_status=ExtractionStatus.PARTIAL if incomplete else ExtractionStatus.SUCCESS,
        ))
    return FinancialExtractionOutput(statements=reports)


def qualitative_result(*, empty=False):
    if empty:
        return QualitativeAnalysisResult(
            analysis_question="demand", reporting_periods_analyzed=[],
            limitations=["No relevant transcript passages were retrieved."],
        )
    source = SourceReference(
        source_id="S001", document_name="CC-Oct26tcs.pdf", reporting_period="Q2 FY27",
        page_number=20, speaker="K Krithivasan", retrieval_score=0.82,
    )
    return QualitativeAnalysisResult(
        analysis_question="demand", reporting_periods_analyzed=["Q2 FY27"],
        management_sentiment=[ManagementSentiment(
            sentiment="positive", explanation="Management described demand as stable.",
            supporting_evidence=["demand remains stable"], source_references=[source],
        )],
        sources=[source],
    )


class FakeQualitativeTool:
    def __init__(self, result=None, error=None):
        self.result = result if result is not None else qualitative_result()
        self.error = error
        self.periods = None

    def analyze(self, question, *, reporting_periods=None):
        self.periods = reporting_periods
        if self.error:
            raise self.error
        return self.result


class FakeLLM:
    def __init__(self, response=None, error=None):
        self.response = response if response is not None else {"cross_analysis": []}
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


def success_reasoning():
    return {"cross_analysis": [{
        "finding": "Q2 FY27 consolidated revenue is supported alongside management's demand commentary.",
        "relationship": "agreement",
        "financial_source_ids": ["F0001"],
        "transcript_source_ids": ["S001"],
    }]}


def make_agent(*, fin=None, fin_error=None, qual=None, qual_error=None, llm=None):
    def extractor():
        if fin_error:
            raise fin_error
        return fin if fin is not None else financial_result()

    return FinancialAnalysisAgent(
        financial_extractor=extractor,
        qualitative_tool=qual or FakeQualitativeTool(error=qual_error),
        llm=llm or FakeLLM(),
    )


def test_both_tools_succeed_preserves_outputs_and_source_provenance():
    financial = financial_result()
    qualitative = qualitative_result()
    llm = FakeLLM(success_reasoning())
    result = make_agent(fin=financial, qual=FakeQualitativeTool(qualitative), llm=llm).analyze(
        "TCS", "Relate revenue to management's demand commentary",
    )
    assert isinstance(result, FinancialAnalysisResult)
    assert result.tool_execution_status.overall_status == "success"
    assert result.financial_analysis == financial
    assert result.qualitative_analysis == qualitative
    finding = result.cross_analysis[0]
    assert finding.financial_sources[0].source_document == "report-Q2 FY27-CONSOLIDATED.pdf"
    assert finding.financial_sources[0].source_page == 5
    assert finding.financial_sources[0].statement_basis == StatementBasis.CONSOLIDATED
    assert finding.transcript_sources[0].document_name == "CC-Oct26tcs.pdf"
    assert finding.transcript_sources[0].page_number == 20
    assert finding.transcript_sources[0].speaker == "K Krithivasan"
    assert llm.structured_kwargs == {"method": "json_schema", "strict": False}


def test_financial_tool_failure_returns_qualitative_partial_result():
    llm = FakeLLM({"cross_analysis": [{
        "finding": "Transcript commentary is available, but comparable financial evidence is missing.",
        "relationship": "insufficient_evidence", "transcript_source_ids": ["S001"],
    }]})
    result = make_agent(fin_error=OSError("source read failed"), llm=llm).analyze("TCS", "demand")
    assert result.financial_analysis is None
    assert result.qualitative_analysis is not None
    assert result.tool_execution_status.overall_status == "partial_success"
    assert result.tool_execution_status.financial_extraction == "failed"


def test_qualitative_failure_returns_financial_partial_result():
    result = make_agent(qual_error=ValueError("retriever unavailable")).analyze("TCS", "revenue")
    assert result.financial_analysis is not None
    assert result.qualitative_analysis is None
    assert result.tool_execution_status.overall_status == "partial_success"
    assert result.tool_execution_status.qualitative_analysis == "failed"


def test_both_specialized_tools_failing_returns_clear_failure_without_master_call():
    llm = FakeLLM()
    result = make_agent(fin_error=OSError(), qual_error=ValueError(), llm=llm).analyze("TCS", "outlook")
    assert result.tool_execution_status.overall_status == "failed"
    assert result.financial_analysis is None and result.qualitative_analysis is None
    assert result.tool_execution_status.master_reasoning == "failed"
    assert llm.calls == 0


@pytest.mark.parametrize(("exception", "category"), [
    (type("AuthenticationError", (Exception,), {})(), "authentication"),
    (type("RateLimitError", (Exception,), {})(), "rate_limit"),
    (type("APITimeoutError", (Exception,), {})(), "timeout"),
])
def test_qualitative_groq_failures_are_safe_and_do_not_trigger_duplicate_master_call(exception, category):
    llm = FakeLLM()
    qual = FakeQualitativeTool(error=QualitativeAnalysisError("redacted failure", category=category))
    result = make_agent(qual=qual, llm=llm).analyze("TCS", "revenue")
    assert result.financial_analysis is not None
    assert result.tool_execution_status.overall_status == "partial_success"
    assert result.tool_execution_status.qualitative_analysis == "failed"
    expected = {"authentication": "authentication", "rate_limit": "rate limit", "timeout": "timed out"}[category]
    assert expected in result.tool_execution_status.errors["qualitative_analysis"].lower()
    assert llm.calls == 0


def test_master_authentication_failure_preserves_both_specialist_outputs():
    error = type("AuthenticationError", (Exception,), {})()
    result = make_agent(llm=FakeLLM(error=error)).analyze("TCS", "demand and revenue")
    assert result.financial_analysis is not None and result.qualitative_analysis is not None
    assert result.cross_analysis == []
    assert result.tool_execution_status.master_reasoning == "failed"
    assert result.tool_execution_status.overall_status == "partial_success"
    assert "GROQ_API_KEY" in result.tool_execution_status.errors["master_reasoning"]


@pytest.mark.parametrize(("exception_type", "message_fragment"), [
    ("AuthenticationError", "authentication failed"),
    ("RateLimitError", "rate limit reached"),
    ("APITimeoutError", "timed out"),
])
def test_master_provider_error_categories_are_sanitized(exception_type, message_fragment):
    error = type(exception_type, (Exception,), {})()
    result = make_agent(llm=FakeLLM(error=error)).analyze("TCS", "demand and revenue")
    message = result.tool_execution_status.errors["master_reasoning"].lower()
    assert message_fragment in message
    assert result.tool_execution_status.overall_status == "partial_success"


def test_invalid_master_json_and_unknown_references_preserve_specialist_outputs():
    malformed = make_agent(llm=FakeLLM("not json")).analyze("TCS", "question")
    assert malformed.tool_execution_status.master_reasoning == "failed"
    assert malformed.tool_execution_status.overall_status == "partial_success"

    fabricated = make_agent(llm=FakeLLM({"cross_analysis": [{
        "finding": "Invented relationship.", "relationship": "context",
        "financial_source_ids": ["F9999"],
    }]})).analyze("TCS", "question")
    assert fabricated.cross_analysis == []
    assert "not returned" in fabricated.tool_execution_status.errors["master_reasoning"]


def test_incomplete_metrics_are_null_and_reported_as_data_quality_notes():
    result = make_agent(fin=financial_result(incomplete=True)).analyze("TCS", "revenue")
    assert result.financial_analysis.statements[0].financial_metrics["net_profit"].value is None
    assert any("unavailable metrics" in note for note in result.data_quality_notes)


def test_reporting_period_discovery_and_empty_transcript_retrieval_are_preserved():
    qual = FakeQualitativeTool(qualitative_result(empty=True))
    result = make_agent(qual=qual).analyze("TCS", "outlook")
    assert set(qual.periods) == {"Q4 FY2025-26", "Q2 FY27"}
    assert result.qualitative_analysis.limitations
    assert "No relevant transcript" in result.data_quality_notes[0]


def test_missing_reporting_period_is_reported_without_inventing_one():
    result = make_agent(fin=financial_result(missing_period=True)).analyze("TCS", "revenue")
    assert any("Reporting period is unidentified" in note for note in result.data_quality_notes)
    assert "UNKNOWN" not in result.reporting_periods


def test_financial_catalog_keeps_basis_and_annual_quarterly_records_separate():
    llm = FakeLLM(success_reasoning())
    result = make_agent(llm=llm).analyze("TCS", "revenue")
    sent_payload = json.loads(llm.messages[1][1])
    statements = sent_payload["financial_tool_output"]["statements"]
    assert {(s["report_period_type"], s["statement_basis"]) for s in statements} == {
        ("QUARTERLY", "CONSOLIDATED"), ("QUARTERLY", "STANDALONE"), ("ANNUAL", "CONSOLIDATED"),
    }
    with pytest.raises(ValidationError, match="cannot mix consolidated and standalone"):
        agent_module.CrossAnalysisFinding(
            finding="mixed", relationship="context",
            financial_sources=[result.cross_analysis[0].financial_sources[0],
                               agent_module._financial_catalog(result.financial_analysis)["F0003"]],
        )
    with pytest.raises(ValidationError, match="cannot mix quarterly and annual"):
        agent_module.CrossAnalysisFinding(
            finding="mixed periods", relationship="context",
            financial_sources=[result.cross_analysis[0].financial_sources[0],
                               agent_module._financial_catalog(result.financial_analysis)["F0005"]],
        )


def test_unsupported_numerical_master_forecast_is_rejected():
    llm = FakeLLM({"cross_analysis": [{
        "finding": "Management commentary means revenue will grow 99% next quarter.",
        "relationship": "agreement", "financial_source_ids": ["F0001"],
        "transcript_source_ids": ["S001"],
    }]})
    result = make_agent(llm=llm).analyze("TCS", "forecast")
    assert result.cross_analysis == []
    assert "numerical claim" in result.tool_execution_status.errors["master_reasoning"].lower()


def test_status_model_rejects_inconsistent_overall_status():
    from app.schemas.analysis import ToolExecutionStatus
    with pytest.raises(ValidationError):
        ToolExecutionStatus(
            financial_extraction="success", qualitative_analysis="success",
            master_reasoning="failed", overall_status="success",
        )


def test_groq_client_is_constructed_without_exposing_credential_in_repr():
    sentinel = "never-print-this-test-secret"
    client = create_groq_client(Settings(groq_api_key=sentinel, llm_model="openai/gpt-oss-120b"))
    assert client.model_name == "openai/gpt-oss-120b"
    assert sentinel not in repr(client)
    assert sentinel not in repr(Settings(groq_api_key=sentinel))


@pytest.mark.parametrize(("status", "expected"), [
    (400, "request_rejected"),
    (413, "context_limit"),
    (429, "rate_limit"),
    (503, "provider_error"),
])
def test_groq_http_statuses_are_classified_without_reading_error_text(status, expected):
    error_type = type("ProviderHTTPError", (Exception,), {"status_code": status})
    error = error_type("secret provider response must not be surfaced")
    assert classify_groq_exception(error) == expected


def test_groq_request_logs_safe_status_only(caplog):
    from app.config.llm import GroqRequestError

    class FailingLLM:
        def invoke(self, _messages):
            error_type = type("ProviderHTTPError", (Exception,), {"status_code": 503})
            raise error_type("secret provider response must not be logged")

    with pytest.raises(GroqRequestError) as raised:
        invoke_structured(FailingLLM(), dict, [("human", "test")])
    assert raised.value.category == "provider_error"
    assert raised.value.status_code == 503
    assert "http_status=503" in caplog.text
    assert "secret provider response" not in caplog.text
