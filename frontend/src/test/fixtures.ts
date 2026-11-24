import type { ForecastAPIResponse, HealthResponse } from "../types/api";

// Fictional test-only fixture. It is never used by the running application.

export const healthyResponse: HealthResponse = {
  status: "healthy",
  application: "TCS Financial Forecasting Agent",
  database_status: "connected",
};

const unavailable = {
  direction: "uncertain" as const,
  evidence_status: "unavailable" as const,
  outlook: "No supported outlook was returned for this section.",
  source_references: [],
};

const financialReference = {
  source: {
    source_id: "FTEST001",
    metric_name: "revenue",
    reporting_period: "Example prior period",
    report_period_type: "QUARTERLY" as const,
    statement_basis: "CONSOLIDATED" as const,
    value: "123.45",
    unit: "SAMPLE_UNITS",
    currency: "XYZ",
    source_document: "synthetic-example-report.pdf",
    source_page: 1,
  },
  supporting_evidence: ["Synthetic UI test evidence only."],
};

export const forecastResponse: ForecastAPIResponse = {
  request_id: "123e4567-e89b-12d3-a456-426614174000",
  status: "completed",
  forecast: {
    company: "Example Corp",
    forecast_for: "Example next period",
    based_on_reporting_periods: ["Example prior period"],
    overall_business_outlook: {
      direction: "cautious",
      evidence_status: "limited",
      outlook: "A cautious qualitative outlook is supported by the returned sample evidence.",
      source_references: [financialReference],
    },
    revenue_outlook: {
      direction: "stable",
      evidence_status: "supported",
      outlook: "The sample financial evidence supports a measured revenue view.",
      source_references: [financialReference],
    },
    profitability_outlook: unavailable,
    operating_margin_outlook: unavailable,
    management_outlook: unavailable,
    business_risks: [],
    business_opportunities: [],
    key_growth_drivers: [],
    key_monitoring_indicators: [],
    confidence_level: "low",
    reasoning: [],
    source_references: [financialReference],
    data_limitations: ["Synthetic fixture; no live forecast was generated."],
  },
};

export const emptyForecastResponse: ForecastAPIResponse = {
  ...forecastResponse,
  forecast: {
    ...forecastResponse.forecast,
    source_references: [],
    overall_business_outlook: unavailable,
    revenue_outlook: unavailable,
    profitability_outlook: unavailable,
    operating_margin_outlook: unavailable,
    management_outlook: unavailable,
    data_limitations: [],
  },
};
