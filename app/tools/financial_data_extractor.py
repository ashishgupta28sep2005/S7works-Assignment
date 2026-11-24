"""Extract traceable metrics from the project's processed TCS PDF reports."""

from __future__ import annotations

import json
import logging
import re
from datetime import date, datetime
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Iterable

from app.ingestion.document_classifier import classify_document
from app.ingestion.models import DocumentType, ExtractedPage, ExtractedTable, ProcessedDocument
from app.ingestion.pdf_loader import PDFLoadError, load_pdf
from app.schemas.financial import (
    ExtractionStatus,
    FinancialExtractionOutput,
    FinancialMetric,
    FinancialStatement,
    MetricComparison,
    MetricStatus,
    QuarterlyComparison,
    StatementBasis,
)

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "data" / "financial_reports"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "financial_extractions"

_NUMBER = re.compile(r"(?<![A-Za-z])\(?-?\d[\d,]*(?:\.\d+)?\)?|(?<!\w)[-–—](?!\w)")
_DATE = re.compile(r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(20\d{2})\b", re.I)
_METRIC_LABELS: dict[str, tuple[re.Pattern[str], ...]] = {
    "revenue": (re.compile(r"^revenue from operations$", re.I), re.compile(r"^revenue$", re.I)),
    "total_expenses": (re.compile(r"^total expenses$", re.I),),
    "operating_profit": (re.compile(r"^(?:operating profit|profit from operations)$", re.I),),
    "operating_margin": (re.compile(r"^(?:operating profit margin|operating margin|opm)$", re.I),),
    "net_profit": (re.compile(r"^profit for (?:the )?(?:period|year)$", re.I), re.compile(r"^net profit$", re.I)),
    "eps": (re.compile(r"^earnings\s*per\s*equity\s*share.*", re.I), re.compile(r"^basic and diluted.*", re.I)),
    "other_income": (re.compile(r"^other income$", re.I),),
    "interest": (re.compile(r"^(?:interest|finance costs?)$", re.I),),
    "depreciation": (re.compile(r"^depreciation and amortisation expense$", re.I),),
    "profit_before_tax": (re.compile(r"^profit before tax$", re.I),),
    "tax": (re.compile(r"^total tax expense$", re.I),),
    "dividend_per_share": (re.compile(r"^total dividend (?:per share|on equity shares).*", re.I),),
    "equity_dividend_percentage": (re.compile(r"^total equity dividend percentage$", re.I),),
}
_CORE_METRICS = ("revenue", "total_expenses", "operating_profit", "operating_margin", "net_profit", "eps")


class FinancialDataExtractorTool:
    """Reusable deterministic tool interface for ingestion and agent phases."""

    name = "financial_data_extractor"
    description = (
        "Extract source-grounded financial statement metrics from processed PDF documents. "
        "Returns separate consolidated and standalone statements with reporting periods and page provenance."
    )

    def run(
        self,
        reports_dir: Path = REPORTS_DIR,
        output_dir: Path = OUTPUT_DIR,
    ) -> FinancialExtractionOutput:
        """Extract the configured report corpus and persist its regular Phase 3 output."""
        return run_extraction(reports_dir=reports_dir, output_dir=output_dir)

    def extract_document(self, document: ProcessedDocument) -> list[FinancialStatement]:
        return extract_financial_document(document)

    def extract_documents(self, documents: Iterable[ProcessedDocument]) -> FinancialExtractionOutput:
        return extract_financial_documents(documents)

    def compare_quarters(
        self,
        statements: Iterable[FinancialStatement],
        from_period: str = "Q1 FY27",
        to_period: str = "Q2 FY27",
        basis: StatementBasis = StatementBasis.CONSOLIDATED,
    ) -> list[QuarterlyComparison]:
        return compare_quarterly_statements(statements, from_period, to_period, basis)


def normalize_financial_number(raw: str) -> Decimal | None:
    """Parse Indian-grouped values, decimals, dashes, and accounting negatives."""
    token = raw.strip().replace("₹", "").replace("INR", "").strip()
    if token in {"-", "–", "—"}:
        return Decimal("0")
    negative = token.startswith("(") and token.endswith(")")
    token = token.strip("() ").replace(",", "")
    try:
        value = Decimal(token)
    except (InvalidOperation, ValueError):
        return None
    return -abs(value) if negative else value


def extract_financial_document(document: ProcessedDocument) -> list[FinancialStatement]:
    """Extract one record per identifiable statement basis; never mix bases."""
    document = classify_document(document)
    if document.metadata.document_type != DocumentType.FINANCIAL_REPORT:
        return []
    candidates = _statement_pages(document)
    if not candidates:
        return [_empty_statement(document, StatementBasis.UNKNOWN, [
            "No recognizable profit-and-loss statement table was found."
        ])]

    statements: list[FinancialStatement] = []
    annual_standalone_is_quarter_only = False
    for page, table, basis in candidates:
        if document.metadata.report_period_type == "ANNUAL" and basis == StatementBasis.STANDALONE:
            lower = page.text.lower()
            heading = lower.find("audited standalone statement of financial results")
            if heading >= 0:
                statement_section = lower[heading:]
                first_metric = statement_section.find("revenue from operations")
                statement_header = statement_section[:first_metric] if first_metric >= 0 else statement_section[:500]
                if "three months ended" in statement_header and "year ended" not in statement_header:
                    annual_standalone_is_quarter_only = True
                    continue
        statement = _extract_statement(document, page, table, basis)
        if not any(
            old.reporting_period == statement.reporting_period
            and old.statement_basis == statement.statement_basis
            for old in statements
        ):
            statements.append(statement)
    if annual_standalone_is_quarter_only:
        unavailable = _empty_statement(document, StatementBasis.STANDALONE, [
            "Standalone statement in this annual report contains quarter-only figures; it was not reinterpreted as FY26 data."
        ])
        unavailable.extraction_status = ExtractionStatus.REVIEW
        statements.append(unavailable)
    return statements


def extract_financial_documents(documents: Iterable[ProcessedDocument]) -> FinancialExtractionOutput:
    statements: list[FinancialStatement] = []
    for document in documents:
        statements.extend(extract_financial_document(document))
    statements.sort(key=lambda item: (_period_sort_key(item.reporting_period), item.statement_basis.value))
    comparisons = compare_quarterly_statements(statements)
    return FinancialExtractionOutput(statements=statements, quarterly_comparisons=comparisons)


def compare_quarterly_statements(
    statements: Iterable[FinancialStatement],
    from_period: str = "Q1 FY27",
    to_period: str = "Q2 FY27",
    basis: StatementBasis = StatementBasis.CONSOLIDATED,
) -> list[QuarterlyComparison]:
    """Compare only matching quarterly periods, units, and statement bases."""
    items = [s for s in statements if s.statement_basis == basis and s.report_period_type == "QUARTERLY"]
    before = next((s for s in items if s.reporting_period == from_period), None)
    after = next((s for s in items if s.reporting_period == to_period), None)
    if before is None or after is None:
        return []
    changes: dict[str, MetricComparison] = {}
    notes: list[str] = []
    for name in ("revenue", "net_profit", "operating_margin"):
        first = before.financial_metrics[name]
        second = after.financial_metrics[name]
        if first.value is None or second.value is None:
            notes.append(f"{name} comparison omitted because one or both source values are unavailable.")
            continue
        if first.unit != second.unit or first.currency != second.currency:
            notes.append(f"{name} comparison omitted because units or currency differ.")
            continue
        value = second.value - first.value
        changes[f"{name}_change"] = MetricComparison(
            name=f"{name} change", value=value, unit=first.unit, currency=first.currency,
            formula=f"{to_period} value - {from_period} value",
            source_documents=[before.source_document, after.source_document],
            source_pages=[p for p in (first.source_page, second.source_page) if p is not None],
        )
        if name in {"revenue", "net_profit"} and first.value != 0:
            growth = (value / abs(first.value) * Decimal("100")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            changes[f"{name}_growth_percent"] = MetricComparison(
                name=f"{name} quarter-over-quarter growth", value=growth, unit="PERCENT", currency=None,
                formula=f"(({to_period} - {from_period}) / abs({from_period})) * 100",
                source_documents=[before.source_document, after.source_document],
                source_pages=[p for p in (first.source_page, second.source_page) if p is not None],
            )
    return [QuarterlyComparison(
        from_period=from_period, to_period=to_period, statement_basis=basis,
        metrics=changes, notes=notes,
    )]


def _statement_pages(
    document: ProcessedDocument,
) -> list[tuple[ExtractedPage, ExtractedTable, StatementBasis]]:
    output = []
    for page in document.pages:
        lower = page.text.lower()
        if "revenue from operations" not in lower:
            continue
        if not any(term in lower for term in ("statement of financial results", "statement of profit", "profit before tax")):
            continue
        tables = [table for table in document.tables if table.page_number == page.page_number]
        table = next((
            t for t in tables
            if "revenue from operations" in _table_text(t).lower()
            or _has_period_header(t)
        ), None)
        if table is None:
            continue
        has_consolidated = "consolidated" in lower
        has_standalone = "standalone" in lower
        if has_consolidated == has_standalone:
            basis = StatementBasis.UNKNOWN
        elif has_consolidated:
            basis = StatementBasis.CONSOLIDATED
        else:
            basis = StatementBasis.STANDALONE
        output.append((page, table, basis))
    return output


def _extract_statement(
    document: ProcessedDocument,
    page: ExtractedPage,
    table: ExtractedTable,
    basis: StatementBasis,
) -> FinancialStatement:
    quarterly = document.metadata.report_period_type != "ANNUAL"
    headers = _period_columns(table)
    selected_index, selected_date = _select_current_column(document, headers, quarterly)
    reporting_period = _period_from_date(selected_date, quarterly) if selected_date else _report_period(document)
    comparative_index, comparative_date = _select_comparative_column(selected_date, headers, quarterly)
    comparative_period = _period_from_date(comparative_date, quarterly) if comparative_date else None
    scale = _unit_scale(page.text)
    metrics: dict[str, FinancialMetric] = {}
    validation_errors: list[str] = []
    unavailable: list[str] = []
    if selected_index is None:
        validation_errors.append("The requested reporting-period column could not be identified; no figures were assigned to it.")

    for metric_name, patterns in _METRIC_LABELS.items():
        matches = _find_metric_lines(page.text, patterns)
        if not matches:
            metrics[metric_name] = FinancialMetric(
                name=metric_name, reporting_period=reporting_period, statement_basis=basis,
                extraction_status=MetricStatus.NOT_FOUND,
            )
            unavailable.append(metric_name)
            continue
        line_index, label = matches[0]
        errors: list[str] = []
        if len(matches) > 1:
            errors.append(f"Multiple candidate rows found for {metric_name}; first exact statement row selected.")
            validation_errors.extend(errors)
        value_tokens = _values_after_line(page.text.splitlines(), line_index)
        chosen = _value_at(value_tokens, selected_index)
        if metric_name == "eps" and not quarterly and selected_index is not None and chosen is None and len(value_tokens) == 2:
            # Some annual-result PDFs render only the two year-ended EPS cells in page text.
            chosen = value_tokens[0]
            comparative_index = 1
        if chosen is None:
            errors.append(f"Could not map a numeric value for {metric_name} to the selected reporting period.")
            validation_errors.extend(errors)
        comparative = _value_at(value_tokens, comparative_index) if comparative_index is not None else None
        unit = "PERCENT" if metric_name in {"operating_margin", "equity_dividend_percentage"} else "INR_PER_SHARE" if metric_name == "eps" or metric_name == "dividend_per_share" else scale
        raw = chosen[1] if chosen else None
        comp_raw = comparative[1] if comparative else None
        metrics[metric_name] = FinancialMetric(
            name=metric_name,
            value=chosen[0] if chosen else None,
            original_value=raw,
            unit=unit if chosen else None,
            currency="INR" if chosen and unit != "PERCENT" else None,
            scale=scale if chosen and unit not in {"PERCENT", "INR_PER_SHARE"} else unit if chosen else None,
            reporting_period=reporting_period,
            comparative_value=comparative[0] if comparative else None,
            comparative_original_value=comp_raw,
            comparative_period=comparative_period if comparative else None,
            source_document=document.metadata.original_filename if chosen else None,
            source_page=page.page_number if chosen else None,
            source_row_label=label if chosen else None,
            statement_basis=basis if chosen else None,
            extraction_status=MetricStatus.EXTRACTED if chosen and not errors else MetricStatus.REVIEW if errors else MetricStatus.NOT_FOUND,
            validation_errors=errors,
        )

    for source_name, growth_name in (("revenue", "revenue_growth"), ("net_profit", "net_profit_growth")):
        source = metrics[source_name]
        growth_value = None
        if source.value is not None and source.comparative_value not in (None, Decimal("0")):
            growth_value = ((source.value - source.comparative_value) / abs(source.comparative_value) * Decimal("100")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        metrics[growth_name] = FinancialMetric(
            name=f"{growth_name}_year_over_year", value=growth_value,
            original_value=None,
            unit="PERCENT" if growth_value is not None else None,
            reporting_period=reporting_period,
            comparative_value=source.comparative_value,
            comparative_period=source.comparative_period,
            source_document=document.metadata.original_filename if growth_value is not None else None,
            source_page=page.page_number if growth_value is not None else None,
            statement_basis=basis if growth_value is not None else None,
            extraction_status=MetricStatus.CALCULATED if growth_value is not None else MetricStatus.NOT_FOUND,
            is_calculated=growth_value is not None,
            formula="((current value - same-period prior-year value) / abs(same-period prior-year value)) * 100",
            validation_errors=[] if growth_value is not None else ["Comparable prior-year figure is unavailable or zero."],
        )

    missing_core = [name for name in _CORE_METRICS if metrics[name].value is None]
    if basis == StatementBasis.UNKNOWN:
        validation_errors.append("Statement basis is not identifiable from this page; review required.")
    if missing_core:
        validation_errors.append("Unavailable core metrics: " + ", ".join(missing_core))
    status = ExtractionStatus.REVIEW if basis == StatementBasis.UNKNOWN or selected_index is None or any(m.extraction_status == MetricStatus.REVIEW for m in metrics.values()) else ExtractionStatus.PARTIAL if missing_core else ExtractionStatus.SUCCESS
    return FinancialStatement(
        source_document=document.metadata.original_filename,
        reporting_period=reporting_period,
        report_period_type="QUARTERLY" if quarterly else "ANNUAL",
        statement_basis=basis,
        currency="INR" if scale else None,
        unit=scale,
        financial_metrics=metrics,
        extraction_status=status,
        validation_errors=validation_errors,
        unrecognized_metrics=unavailable,
    )


def _find_metric_lines(text: str, patterns: tuple[re.Pattern[str], ...]) -> list[tuple[int, str]]:
    output = []
    for index, raw_line in enumerate(text.splitlines()):
        line = _clean_label(raw_line)
        if not line:
            continue
        for pattern in patterns:
            if pattern.fullmatch(line):
                output.append((index, line))
                break
    return output


def _values_after_line(
    lines: list[str], start: int, max_lines: int = 7,
) -> list[tuple[Decimal, str]]:
    values: list[tuple[Decimal, str]] = []
    for offset, line in enumerate(lines[start:start + max_lines + 1]):
        cleaned = _clean_label(line)
        if not cleaned:
            continue
        if offset > 0 and _looks_like_metric_label(cleaned):
            break
        tokens = _NUMBER.findall(line)
        for token in tokens:
            if token in {"-", "–", "—"} and cleaned not in {"-", "–", "—"}:
                continue
            value = normalize_financial_number(token)
            if value is not None:
                values.append((value, token))
    return values


def _looks_like_metric_label(line: str) -> bool:
    normalized = line.lower().strip().rstrip(":")
    if normalized in {"expenses", "tax expense", "exceptional items", "other comprehensive income (oci)"}:
        return True
    return any(pattern.fullmatch(normalized) for patterns in _METRIC_LABELS.values() for pattern in patterns)


def _period_columns(table: ExtractedTable) -> list[tuple[int, str, str]]:
    """Return (zero-based numeric column, group, date) from table heading rows."""
    if len(table.rows) < 2:
        return []
    groups, dates = table.rows[0], table.rows[1]
    output = []
    current_group = ""
    for column in range(1, min(len(groups), len(dates))):
        if groups[column]:
            current_group = groups[column] or current_group
        date_text = dates[column]
        if date_text:
            output.append((column - 1, current_group.lower(), date_text))
    return output


def _has_period_header(table: ExtractedTable) -> bool:
    if len(table.rows) < 3:
        return False
    header_text = " ".join((cell or "") for cell in table.rows[0])
    date_text = " ".join((cell or "") for cell in table.rows[1])
    return bool(re.search(r"(?:three months|year) ended", header_text, re.I) and _DATE.search(date_text))


def _select_current_column(
    document: ProcessedDocument,
    headers: list[tuple[int, str, str]],
    quarterly: bool,
) -> tuple[int, str | None]:
    groups = [(i, group, label, _parse_date(label)) for i, group, label in headers]
    target_date = _parse_date(document.metadata.reporting_period or "")
    preferred = [item for item in groups if ("three months ended" in item[1] if quarterly else "year ended" in item[1])]
    if target_date:
        matching = [item for item in preferred if item[3] == target_date]
        if matching:
            return matching[0][0], matching[0][2]
    if preferred and target_date is None:
        return preferred[0][0], preferred[0][2]
    return (None, None)


def _select_comparative_column(
    current_label: str | None,
    headers: list[tuple[int, str, str]],
    quarterly: bool,
) -> tuple[int | None, str | None]:
    current_date = _parse_date(current_label or "")
    if current_date is None:
        return None, None
    try:
        prior = current_date.replace(year=current_date.year - 1)
    except ValueError:
        return None, None
    group_needed = "three months ended" if quarterly else "year ended"
    for index, group, label in headers:
        if group_needed in group and _parse_date(label) == prior:
            return index, label
    return None, None


def _value_at(values: list[tuple[Decimal, str]], index: int | None) -> tuple[Decimal, str] | None:
    return values[index] if index is not None and index < len(values) else None


def _parse_date(text: str) -> date | None:
    match = _DATE.search(text)
    if not match:
        return None
    try:
        return date(int(match.group(3)), _month_number(match.group(1)), int(match.group(2)))
    except ValueError:
        return None


def _month_number(month: str) -> int:
    return datetime.strptime(f"01 {month} 2000", "%d %B %Y").month


def _period_from_date(label: str | None, quarterly: bool) -> str:
    parsed = _parse_date(label or "")
    if not parsed:
        return label or "UNKNOWN"
    if not quarterly:
        return f"FY{str(parsed.year)[-2:]}"
    quarter = ((parsed.month - 4) % 12) // 3 + 1
    fy = parsed.year + 1 if parsed.month >= 4 else parsed.year
    return f"Q{quarter} FY{str(fy)[-2:]}"


def _report_period(document: ProcessedDocument) -> str:
    label = document.metadata.reporting_period or "UNKNOWN"
    parsed = _parse_date(label)
    return _period_from_date(label, document.metadata.report_period_type != "ANNUAL") if parsed else label


def _unit_scale(text: str) -> str | None:
    lowered = text.lower()
    if "crore" in lowered:
        return "CRORES"
    if "lakh" in lowered:
        return "LAKHS"
    if "million" in lowered:
        return "MILLIONS"
    return None


def _clean_label(text: str) -> str:
    return " ".join(text.replace("\u00a0", " ").split()).strip().rstrip(":")


def _table_text(table: ExtractedTable) -> str:
    return " ".join(cell or "" for row in table.rows for cell in row)


def _empty_statement(
    document: ProcessedDocument, basis: StatementBasis, errors: list[str],
) -> FinancialStatement:
    period = _report_period(document)
    metrics = {
        name: FinancialMetric(name=name, reporting_period=period, statement_basis=basis,
                              extraction_status=MetricStatus.NOT_FOUND)
        for name in (*_METRIC_LABELS.keys(), "revenue_growth", "net_profit_growth")
    }
    return FinancialStatement(
        source_document=document.metadata.original_filename, reporting_period=period,
        report_period_type="ANNUAL" if document.metadata.report_period_type == "ANNUAL" else "QUARTERLY",
        statement_basis=basis, currency=None, unit=None, financial_metrics=metrics,
        extraction_status=ExtractionStatus.FAILED, validation_errors=errors,
        unrecognized_metrics=list(_METRIC_LABELS),
    )


def _period_sort_key(period: str) -> tuple[int, int]:
    match = re.fullmatch(r"Q([1-4]) FY(\d{2})", period)
    if match:
        return int(match.group(2)), int(match.group(1))
    year = re.fullmatch(r"FY(\d{2})", period)
    return (int(year.group(1)), 5) if year else (0, 0)


def run_extraction(reports_dir: Path = REPORTS_DIR, output_dir: Path = OUTPUT_DIR) -> FinancialExtractionOutput:
    documents: list[ProcessedDocument] = []
    errors: list[str] = []
    if not reports_dir.is_dir():
        errors.append(f"Financial reports directory does not exist: {reports_dir}")
    else:
        for path in sorted(reports_dir.glob("*.pdf")):
            try:
                document = classify_document(load_pdf(path))
                if document.metadata.document_type == DocumentType.FINANCIAL_REPORT:
                    documents.append(document)
                else:
                    errors.append(f"Skipped unrecognized/non-financial PDF: {path.name}")
            except (OSError, PDFLoadError, ValueError) as exc:
                logger.exception("Unable to process report %s", path)
                errors.append(f"{path.name}: {exc}")
    output = extract_financial_documents(documents)
    output.errors.extend(errors)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "financial_data.json"
    target.write_text(output.model_dump_json(indent=2), encoding="utf-8")
    return output


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    output = run_extraction()
    print(f"Reports extracted: {len(output.statements)} statements")
    for statement in output.statements:
        missing = ", ".join(statement.unrecognized_metrics) or "none"
        print(
            f"{statement.source_document}: {statement.reporting_period} "
            f"{statement.statement_basis.value}, {statement.extraction_status.value}; "
            f"missing/unrecognized: {missing}"
        )
        for name in ("revenue", "total_expenses", "operating_profit", "operating_margin", "net_profit", "eps"):
            metric = statement.financial_metrics[name]
            if metric.value is not None:
                print(f"  {name}: {metric.value} {metric.unit} (p.{metric.source_page}, {metric.original_value})")
        for name, metric in statement.financial_metrics.items():
            if name not in {"revenue", "total_expenses", "operating_profit", "operating_margin", "net_profit", "eps"} and metric.value is not None:
                print(f"  {name}: {metric.value} {metric.unit} (p.{metric.source_page}, {metric.original_value or metric.formula})")
        for issue in statement.validation_errors:
            print(f"  review: {issue}")
    for comparison in output.quarterly_comparisons:
        print(f"Comparison {comparison.from_period} to {comparison.to_period} ({comparison.statement_basis.value}):")
        for item in comparison.metrics.values():
            print(f"  {item.name}: {item.value} {item.unit} (calculated; pages {item.source_pages})")
    for error in output.errors:
        logger.error(error)
    print(f"JSON written to: {OUTPUT_DIR / 'financial_data.json'}")
    return 1 if output.errors and not output.statements else 0


if __name__ == "__main__":
    raise SystemExit(main())
