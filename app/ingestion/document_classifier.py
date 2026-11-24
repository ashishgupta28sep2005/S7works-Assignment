"""Content-based classification for financial reports and earnings calls."""

from __future__ import annotations

import re
from datetime import datetime

from app.ingestion.models import DocumentType, ProcessedDocument

_QUARTER_PATTERN = re.compile(r"\b(Q[1-4])\s*(?:FY\s*)?((?:20)?\d{2}(?:[-/]\d{2})?)\b", re.I)
_DATE_PATTERNS = (
    re.compile(r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+20\d{2}\b", re.I),
    re.compile(r"\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+20\d{2}\b", re.I),
)
_SPEAKER_PATTERN = re.compile(
    r"\b(?:Mr|Ms|Mrs|Dr)\.\s+([A-Z][A-Za-z.' -]{2,60}?),\s*"
    r"(?:Chief\b|Managing\b|Head(?:\s+of\b|\s*[-–])|Executive\s+Director\b)",
)


def classify_document(document: ProcessedDocument) -> ProcessedDocument:
    """Set document type and fields only when extracted content supports them."""
    text = "\n".join(page.text for page in document.pages if page.text)
    if document.metadata.title:
        text = f"{document.metadata.title}\n{text}"
    normalized = re.sub(r"\s+", " ", text).strip()
    lowered = normalized.lower()
    metadata = document.metadata
    company = _find_company(lowered)

    financial_hits = sum(term in lowered for term in (
        "financial results", "statement of profit and loss", "balance sheet",
        "quarter ended", "consolidated financial", "standalone financial",
    ))
    transcript_hits = sum(term in lowered for term in (
        "earnings conference call", "earnings call transcript", "transcript",
        "operator:", "question-and-answer", "question and answer session",
    ))

    if company and financial_hits >= 2 and financial_hits > transcript_hits:
        metadata.document_type = DocumentType.FINANCIAL_REPORT
        metadata.company = company
        metadata.reporting_period = _find_reporting_period(normalized)
        opening = lowered[:8000]
        has_quarter = bool(re.search(r"quarter ended|interim financial results|three months ended", opening))
        has_annual = bool(re.search(r"annual report|financial results? for (?:the )?year ended", opening))
        if has_quarter and has_annual:
            metadata.report_period_type = "QUARTERLY_AND_ANNUAL"
        elif has_annual:
            metadata.report_period_type = "ANNUAL"
        elif has_quarter:
            metadata.report_period_type = "QUARTERLY"
        else:
            metadata.report_period_type = None
        if "consolidated" in lowered and "standalone" in lowered:
            metadata.consolidation = "BOTH"
        elif "consolidated" in lowered:
            metadata.consolidation = "CONSOLIDATED"
        elif "standalone" in lowered:
            metadata.consolidation = "STANDALONE"
        else:
            metadata.consolidation = None
        return document

    if company and transcript_hits >= 2 and transcript_hits > financial_hits:
        metadata.document_type = DocumentType.EARNINGS_TRANSCRIPT
        metadata.company = company
        metadata.reporting_period = _find_quarter(normalized)
        metadata.earnings_call_date = _find_date(normalized)
        metadata.management_speakers = list(dict.fromkeys(
            match.group(1).strip() for match in _SPEAKER_PATTERN.finditer(text)
        ))
        return document

    metadata.document_type = DocumentType.UNKNOWN
    metadata.company = company
    document.errors.append(
        "Document type is uncertain: extracted content lacks enough company-specific financial-report or earnings-transcript evidence."
    )
    return document


def _find_company(text: str) -> str | None:
    if re.search(r"\btata consultancy services\b|\btcs\b", text):
        return "TCS"
    return None


def _find_reporting_period(text: str) -> str | None:
    quarter = _find_quarter(text)
    if quarter:
        return quarter
    match = re.search(r"(?:quarter|six-month|year)\s+(?:and\s+)?(?:period\s+)?ended\s+(?:on\s+)?([A-Za-z]+\s+\d{1,2},?\s+20\d{2})", text, re.I)
    if match:
        return match.group(1)
    return None


def _find_quarter(text: str) -> str | None:
    match = _QUARTER_PATTERN.search(text)
    if match:
        year = match.group(2).upper().replace("FY", "")
        return f"{match.group(1).upper()} FY{year}"
    full_year = re.search(r"\b(Q[1-4])\s*&\s*Full Year\s+FY\s*((?:20)?\d{2})\b", text, re.I)
    if full_year:
        return f"{full_year.group(1).upper()} FY{full_year.group(2)}"
    return None


def _find_date(text: str) -> str | None:
    # Match the actual transcript heading before the cover-letter publication date.
    call_heading = re.search(
        r"(?:Transcript\s+for\s+Earnings\s+Conference\s+Call|"
        r"Financial\s+Results.{0,100}?Conference\s+Call)\s+"
        r"((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+20\d{2})",
        text,
        re.I,
    )
    if call_heading:
        try:
            return datetime.strptime(call_heading.group(1), "%B %d, %Y").date().isoformat()
        except ValueError:
            pass
    for pattern in _DATE_PATTERNS:
        match = pattern.search(text)
        if match:
            try:
                return datetime.strptime(match.group(), "%B %d, %Y").date().isoformat()
            except ValueError:
                try:
                    return datetime.strptime(match.group(), "%d %B %Y").date().isoformat()
                except ValueError:
                    pass
    return None
