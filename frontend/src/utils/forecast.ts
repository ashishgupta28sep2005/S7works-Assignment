import type { ForecastAPIResponse, SourceReference } from "../types/api";

export function collectSources(response: ForecastAPIResponse): SourceReference[] {
  const forecast = response.forecast;
  const sections = [
    forecast.overall_business_outlook,
    forecast.revenue_outlook,
    forecast.profitability_outlook,
    forecast.operating_margin_outlook,
    forecast.management_outlook,
  ];
  const refs = [
    ...forecast.source_references,
    ...sections.flatMap((section) => section.source_references),
    ...forecast.business_risks.flatMap((item) => item.source_references),
    ...forecast.business_opportunities.flatMap((item) => item.source_references),
    ...forecast.key_growth_drivers.flatMap((item) => item.source_references),
    ...forecast.key_monitoring_indicators.flatMap((item) => item.source_references),
    ...forecast.reasoning.flatMap((item) => item.source_references),
  ];
  const unique = new Map<string, SourceReference>();
  for (const ref of refs) unique.set(ref.source.source_id, ref);
  return [...unique.values()];
}

export function sourceDocument(source: SourceReference["source"]): string {
  return "source_document" in source ? source.source_document : source.document_name;
}

export function sourcePage(source: SourceReference["source"]): number {
  return "source_page" in source ? source.source_page : source.page_number;
}

export function sourcePeriod(source: SourceReference["source"]): string | null {
  return source.reporting_period ?? null;
}

export function sourceSpeaker(source: SourceReference["source"]): string | null {
  return "speaker" in source ? source.speaker ?? null : null;
}

export function sourceRelevance(source: SourceReference["source"]): number | null {
  return "retrieval_score" in source ? source.retrieval_score : null;
}
