"""Integration checks against the three PDFs supplied in the project data folder."""

from decimal import Decimal
from pathlib import Path

import pytest

from app.ingestion.document_classifier import classify_document
from app.ingestion.pdf_loader import load_pdf
from app.schemas.financial import ExtractionStatus, StatementBasis
from app.tools.financial_data_extractor import extract_financial_documents


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_NAMES = ("June-2026TCS.pdf", "Sep-2026TCS.pdf", "March-2026TCS.pdf")


@pytest.fixture(scope="module")
def extracted_reports():
    report_dir = PROJECT_ROOT / "data" / "financial_reports"
    missing = [name for name in REPORT_NAMES if not (report_dir / name).is_file()]
    if missing:
        pytest.skip("Actual report PDFs not present: " + ", ".join(missing))
    documents = [classify_document(load_pdf(report_dir / name)) for name in REPORT_NAMES]
    return extract_financial_documents(documents)


def find_statement(output, period: str, basis: StatementBasis = StatementBasis.CONSOLIDATED):
    return next(s for s in output.statements
                if s.reporting_period == period and s.statement_basis == basis)


def test_actual_consolidated_period_values_and_provenance(extracted_reports) -> None:
    q1 = find_statement(extracted_reports, "Q1 FY27")
    q2 = find_statement(extracted_reports, "Q2 FY27")
    fy26 = find_statement(extracted_reports, "FY26")

    assert (q1.financial_metrics["revenue"].value,
            q1.financial_metrics["revenue"].source_page) == (Decimal("72275"), 8)
    assert (q1.financial_metrics["total_expenses"].value,
            q1.financial_metrics["net_profit"].value,
            q1.financial_metrics["eps"].value) == (
                Decimal("55231"), Decimal("13420"), Decimal("36.90"))
    assert (q2.financial_metrics["revenue"].value,
            q2.financial_metrics["revenue"].source_page) == (Decimal("73188"), 8)
    assert (q2.financial_metrics["total_expenses"].value,
            q2.financial_metrics["net_profit"].value,
            q2.financial_metrics["eps"].value) == (
                Decimal("55887"), Decimal("13934"), Decimal("38.37"))
    assert (fy26.financial_metrics["revenue"].value,
            fy26.financial_metrics["revenue"].source_page) == (Decimal("267021"), 9)
    assert (fy26.financial_metrics["total_expenses"].value,
            fy26.financial_metrics["net_profit"].value,
            fy26.financial_metrics["eps"].value) == (
                Decimal("201410"), Decimal("49454"), Decimal("136.01"))


def test_actual_comparison_and_annual_standalone_guard(extracted_reports) -> None:
    assert len(extracted_reports.quarterly_comparisons) == 1
    comparison = extracted_reports.quarterly_comparisons[0]
    assert comparison.metrics["revenue_change"].value == Decimal("913")
    assert comparison.metrics["net_profit_change"].value == Decimal("514")

    standalone_fy26 = find_statement(extracted_reports, "FY26", StatementBasis.STANDALONE)
    assert standalone_fy26.extraction_status == ExtractionStatus.REVIEW
    assert standalone_fy26.financial_metrics["revenue"].value is None
    assert any("quarter-only" in error for error in standalone_fy26.validation_errors)
