"""Unit tests use synthetic statement-shaped data, never TCS reported figures."""

from decimal import Decimal

import pytest

from app.ingestion.models import (
    DocumentMetadata,
    DocumentType,
    ExtractedPage,
    ExtractedTable,
    ProcessedDocument,
    ProcessingStatus,
)
from app.schemas.financial import (
    ExtractionStatus,
    FinancialMetric,
    FinancialStatement,
    MetricStatus,
    StatementBasis,
)
from app.tools import financial_data_extractor as extractor


def synthetic_document(
    period: str,
    report_type: str,
    dates: list[str],
    values: dict[str, list[str]],
    *,
    group_names: list[str] | None = None,
    basis: str = "Consolidated",
    repeat_revenue: bool = False,
    unit: str = "INR crore",
) -> ProcessedDocument:
    group_names = group_names or ["Three months ended"] + [""] * (len(dates) - 1)
    lines = [f"{unit}", " ".join(group_names), *dates]
    for label, observations in values.items():
        lines.append(label)
        lines.extend(observations)
        if repeat_revenue and label == "Revenue from operations":
            lines.extend(["Revenue", "999"])
    lines.append(f"Audited {basis} Statement of Financial Results")
    table_rows = [
        [None, *group_names],
        [None, *dates],
        ["Revenue from operations", *[" ".join(values.get("Revenue from operations", []))] * len(dates)],
    ]
    return ProcessedDocument(
        metadata=DocumentMetadata(
            original_filename=f"synthetic-{period}.pdf", source_path=f"synthetic-{period}.pdf",
            document_type=DocumentType.FINANCIAL_REPORT,
            company="Example Systems", reporting_period=period, report_period_type=report_type,
        ),
        pages=[ExtractedPage(page_number=7, text="\n".join(lines))],
        tables=[ExtractedTable(page_number=7, table_number=1, rows=table_rows)],
        processing_status=ProcessingStatus.PROCESSED,
    )


@pytest.fixture
def bypass_company_classifier(monkeypatch):
    monkeypatch.setattr(extractor, "classify_document", lambda doc: doc)


def test_normalize_financial_values() -> None:
    assert extractor.normalize_financial_number("1,23,456.75") == Decimal("123456.75")
    assert extractor.normalize_financial_number("(1,234.50)") == Decimal("-1234.50")
    assert extractor.normalize_financial_number("-7.25") == Decimal("-7.25")
    assert extractor.normalize_financial_number("-") == Decimal("0")
    assert extractor.normalize_financial_number("not-a-number") is None


def test_extracts_metrics_provenance_and_comparatives(bypass_company_classifier) -> None:
    doc = synthetic_document(
        "June 30, 2026", "QUARTERLY", ["June 30, 2026", "June 30, 2025"],
        {
            "Revenue from operations": ["1,234.50", "1,000.25"],
            "Other income": ["25", "20"],
            "TOTAL EXPENSES": ["(100.50)", "90.25"],
            "Operating profit": ["200.00", "180.00"],
            "Operating margin": ["16.20%", "18.00%"],
            "Profit before tax": ["160", "150"],
            "Profit for the period": ["-125.75", "110.00"],
            "Earnings per equity share - Basic and diluted": ["3.75", "3.25"],
            "Total dividend on equity shares": ["1.25", "1.00"],
            "Total equity dividend percentage": ["125", "100"],
        },
    )
    statement = extractor.extract_financial_document(doc)[0]
    assert statement.reporting_period == "Q1 FY27"
    assert statement.statement_basis == StatementBasis.CONSOLIDATED
    assert statement.financial_metrics["revenue"].value == Decimal("1234.50")
    assert statement.financial_metrics["revenue"].comparative_value == Decimal("1000.25")
    assert statement.financial_metrics["total_expenses"].value == Decimal("-100.50")
    assert statement.financial_metrics["operating_profit"].value == Decimal("200.00")
    assert statement.financial_metrics["operating_margin"].unit == "PERCENT"
    assert statement.financial_metrics["net_profit"].value == Decimal("-125.75")
    eps = statement.financial_metrics["eps"]
    assert eps.value == Decimal("3.75")
    assert eps.unit == "INR_PER_SHARE"
    assert eps.source_page == 7
    assert eps.source_document == "synthetic-June 30, 2026.pdf"
    assert statement.financial_metrics["dividend_per_share"].value == Decimal("1.25")
    assert statement.financial_metrics["dividend_per_share"].unit == "INR_PER_SHARE"
    assert statement.financial_metrics["equity_dividend_percentage"].unit == "PERCENT"
    assert statement.financial_metrics["revenue_growth"].value == Decimal("23.42")


def test_missing_metrics_are_null_not_zero(bypass_company_classifier) -> None:
    doc = synthetic_document(
        "June 30, 2026", "QUARTERLY", ["June 30, 2026"],
        {"Revenue from operations": ["10"]},
    )
    statement = extractor.extract_financial_document(doc)[0]
    assert statement.financial_metrics["revenue"].value == Decimal("10")
    assert statement.financial_metrics["operating_profit"].value is None
    assert statement.financial_metrics["operating_profit"].extraction_status == MetricStatus.NOT_FOUND
    assert "operating_profit" in statement.unrecognized_metrics
    assert statement.extraction_status == ExtractionStatus.PARTIAL


def test_annual_report_selects_annual_column_not_q4(bypass_company_classifier) -> None:
    doc = synthetic_document(
        "March 31, 2026", "ANNUAL",
        ["March 31, 2026", "December 31, 2025", "March 31, 2025", "March 31, 2026", "March 31, 2025"],
        {"Revenue from operations": ["70", "60", "55", "267", "250"],
         "Profit for the year": ["10", "9", "8", "49", "45"]},
        group_names=["Three months ended", "", "", "Year ended", ""],
    )
    statement = extractor.extract_financial_document(doc)[0]
    assert statement.reporting_period == "FY26"
    assert statement.report_period_type == "ANNUAL"
    assert statement.financial_metrics["revenue"].value == Decimal("267")
    assert statement.financial_metrics["revenue"].comparative_value == Decimal("250")


def test_unknown_statement_basis_requires_review(bypass_company_classifier) -> None:
    doc = synthetic_document(
        "June 30, 2026", "QUARTERLY", ["June 30, 2026"],
        {"Revenue from operations": ["10"]}, basis="Financial",
    )
    statement = extractor.extract_financial_document(doc)[0]
    assert statement.statement_basis == StatementBasis.UNKNOWN
    assert statement.extraction_status == ExtractionStatus.REVIEW
    assert any("basis" in error.lower() for error in statement.validation_errors)


def test_duplicate_metric_rows_flag_for_review(bypass_company_classifier) -> None:
    doc = synthetic_document(
        "June 30, 2026", "QUARTERLY", ["June 30, 2026"],
        {"Revenue from operations": ["10"]}, repeat_revenue=True,
    )
    statement = extractor.extract_financial_document(doc)[0]
    assert statement.financial_metrics["revenue"].value == Decimal("10")
    assert statement.financial_metrics["revenue"].extraction_status == MetricStatus.REVIEW


def test_quarterly_comparison_and_basis_unit_guard() -> None:
    def record(period: str, value: str, basis: StatementBasis = StatementBasis.CONSOLIDATED,
               unit: str = "CRORES") -> FinancialStatement:
        metric = FinancialMetric(
            name="revenue", value=Decimal(value), unit=unit, currency="INR",
            reporting_period=period, source_document=f"{period}.pdf", source_page=3,
            statement_basis=basis, extraction_status=MetricStatus.EXTRACTED,
        )
        missing = FinancialMetric(name="net_profit", reporting_period=period,
                                  extraction_status=MetricStatus.NOT_FOUND)
        margin = FinancialMetric(name="operating_margin", reporting_period=period,
                                 extraction_status=MetricStatus.NOT_FOUND)
        return FinancialStatement(
            source_document=f"{period}.pdf", reporting_period=period,
            report_period_type="QUARTERLY", statement_basis=basis,
            financial_metrics={"revenue": metric, "net_profit": missing, "operating_margin": margin},
            extraction_status=ExtractionStatus.PARTIAL,
        )

    q1, q2 = record("Q1 FY27", "100"), record("Q2 FY27", "110")
    result = extractor.compare_quarterly_statements([q1, q2])[0]
    assert result.metrics["revenue_change"].value == Decimal("10")
    assert result.metrics["revenue_growth_percent"].value == Decimal("10.00")
    assert any("net_profit comparison omitted" in note for note in result.notes)

    mismatched = record("Q2 FY27", "110", unit="MILLIONS")
    assert extractor.compare_quarterly_statements([q1, mismatched])[0].metrics == {}
