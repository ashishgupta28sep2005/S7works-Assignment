"""Mock-only tests; synthetic transcript fragments are not TCS source evidence."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.rag.models import RetrievalHit, RetrievalResponse, TranscriptChunk
from app.schemas.qualitative import LLMQualitativeAnalysis
from app.tools import qualitative_analysis as qa


class FakeRetriever:
    def __init__(self, chunks=()):
        self.chunks = list(chunks)
        self.calls = []

    def retrieve(self, query, *, reporting_period=None, top_k=None):
        self.calls.append((query, reporting_period))
        hits = [
            RetrievalHit(score=0.81, relevant=True, chunk=chunk)
            for chunk in self.chunks if chunk.reporting_period == reporting_period
        ]
        reason = None if hits else "No chunks for test period"
        return RetrievalResponse(query=query, matches=hits, rejected_reason=reason)


class FakeLLM:
    def __init__(self, response):
        self.response = response
        self.messages = None
        self.schema = None
        self.invocations = 0

    def with_structured_output(self, schema, **kwargs):
        self.schema = schema
        return self

    def invoke(self, messages):
        self.messages = messages
        self.invocations += 1
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def chunk(chunk_id, period, text, speaker=None, page=7):
    return TranscriptChunk(
        company="TCS", document_name=f"transcript-{period}.pdf", reporting_period=period,
        page_number=page, speaker=speaker, speaker_role="management", chunk_id=chunk_id, original_text=text,
        document_hash=(chunk_id * 64)[:64],
    )


PASSAGE_OLD = chunk("a", "Q4 FY2025-26", "Management said demand remained resilient.", "CEO")
PASSAGE_NEW = chunk("b", "Q2 FY27", "Management said demand remained resilient and stable.", None, 12)


def base_response(**updates):
    response = {
        "recurring_themes": [],
        "management_sentiment": [],
        "forward_looking_statements": [],
        "business_risks": [],
        "business_opportunities": [],
        "limitations": [],
    }
    response.update(updates)
    return response


def run(response, chunks=(PASSAGE_OLD, PASSAGE_NEW)):
    fake_llm = FakeLLM(response)
    tool = qa.QualitativeAnalysisTool(FakeRetriever(chunks), fake_llm)
    return tool.analyze("Summarize management's demand commentary"), fake_llm


def test_successful_structured_analysis_resolves_trusted_refs_and_missing_speaker():
    result, llm = run(base_response(management_sentiment=[{
        "sentiment": "positive",
        "explanation": "Management described demand as resilient.",
        "supporting_evidence": ["demand remained resilient"],
        "source_references": ["S001"],
    }]))
    assert llm.schema is LLMQualitativeAnalysis
    assert result.management_sentiment[0].source_references[0].document_name == "transcript-Q4 FY2025-26.pdf"
    assert result.management_sentiment[0].source_references[0].speaker == "CEO"
    assert result.sources[0].retrieval_score == 0.81
    assert result.reporting_periods_analyzed == ["Q2 FY27", "Q4 FY2025-26"]
    no_speaker_response = base_response(management_sentiment=[{
        "sentiment": "neutral", "explanation": "Management described demand as stable.",
        "supporting_evidence": ["demand remained resilient and stable"],
        "source_references": ["S001"],
    }])
    no_speaker = qa.QualitativeAnalysisTool(
        FakeRetriever((PASSAGE_NEW,)), FakeLLM(no_speaker_response)
    ).analyze("demand", reporting_periods=["Q2 FY27"])
    assert no_speaker.management_sentiment[0].source_references[0].speaker is None


def test_missing_api_key_is_a_clear_configuration_error(monkeypatch):
    monkeypatch.setattr(qa, "get_settings", lambda: SimpleNamespace(
        llm_provider="groq", groq_api_key=None, llm_model="test-model",
    ))
    with pytest.raises(qa.LLMConfigurationError, match="GROQ_API_KEY"):
        qa.QualitativeAnalysisTool(FakeRetriever())


def test_empty_retrieval_returns_limitation_without_calling_llm():
    llm = FakeLLM(base_response())
    result = qa.QualitativeAnalysisTool(FakeRetriever(), llm).analyze("demand")
    assert llm.invocations == 0
    assert not result.sources
    assert "No relevant transcript passages" in result.limitations[0]


def test_invalid_json_and_schema_validation_fail_clearly():
    with pytest.raises(qa.QualitativeAnalysisError, match="required structured schema"):
        run("not json")
    with pytest.raises(qa.QualitativeAnalysisError, match="required structured schema"):
        run(base_response(management_sentiment=[{
            "sentiment": "very good", "explanation": "Unsupported label.",
            "supporting_evidence": ["demand remained resilient"], "source_references": ["S001"],
        }]))


def test_fabricated_source_reference_is_rejected():
    response = base_response(forward_looking_statements=[{
        "statement": "Management committed to faster growth.", "business_area": "demand",
        "outlook_direction": "positive", "source_references": ["S999"],
    }])
    with pytest.raises(qa.InvalidSourceReferenceError, match="Unsupported source reference"):
        run(response)


def test_recurrence_requires_distinct_reporting_periods_and_quotes_are_checked():
    valid = base_response(recurring_themes=[{
        "theme": "Resilient demand", "description": "Management described demand as resilient.",
        "supporting_evidence": ["demand remained resilient"],
        "source_references": ["S001", "S002"],
    }])
    result, _ = run(valid)
    assert len(result.recurring_themes) == 1
    assert {ref.reporting_period for ref in result.recurring_themes[0].source_references} == {
        "Q4 FY2025-26", "Q2 FY27",
    }

    single_period = base_response(recurring_themes=[{
        "theme": "Resilient demand", "description": "Repeated passages from one period.",
        "supporting_evidence": ["demand remained resilient"], "source_references": ["S001"],
    }])
    result, _ = run(single_period)
    assert result.recurring_themes == []
    assert any("two reporting periods" in note for note in result.limitations)

    fabricated_quote = base_response(management_sentiment=[{
        "sentiment": "positive", "explanation": "Positive.",
        "supporting_evidence": ["demand will triple"], "source_references": ["S001"],
    }])
    result, _ = run(fabricated_quote)
    assert not result.management_sentiment
    assert any("did not exactly match cited text" in note for note in result.limitations)


def test_analyst_question_guardrails_and_no_unsupported_numerical_forecast():
    llm = FakeLLM(base_response())
    qa.QualitativeAnalysisTool(FakeRetriever((PASSAGE_OLD,)), llm).analyze(
        "demand", reporting_periods=["Q4 FY2025-26"],
    )
    assert "Analyst questions are not management commitments" in llm.messages[0][1]

    response = base_response(forward_looking_statements=[{
        "statement": "Revenue will grow 25% next year.", "business_area": "revenue",
        "outlook_direction": "positive", "source_references": ["S001"],
    }])
    with pytest.raises(qa.InvalidSourceReferenceError, match="Numerical claim '25%'"):
        run(response)


def test_analyst_question_is_not_included_as_management_evidence():
    analyst_question = chunk(
        "c", "Q4 FY2025-26", "Will you commit to 50% revenue growth next year?", "Analyst",
    ).model_copy(update={"speaker_role": "analyst"})
    llm = FakeLLM(base_response())
    result = qa.QualitativeAnalysisTool(
        FakeRetriever((analyst_question, PASSAGE_OLD)), llm,
    ).analyze("What growth guidance did management give?", reporting_periods=["Q4 FY2025-26"])
    assert not result.forward_looking_statements
    evidence_prompt = llm.messages[1][1]
    assert "Will you commit to 50%" not in evidence_prompt
    assert "demand remained resilient" in evidence_prompt


def test_insufficient_evidence_limitations_are_preserved():
    result, _ = run(base_response(limitations=["No reliable guidance found in the retrieved passages."]))
    assert result.limitations == ["No reliable guidance found in the retrieved passages."]

