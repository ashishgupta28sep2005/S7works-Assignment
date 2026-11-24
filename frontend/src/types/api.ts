import { z } from "zod";

const decimalSchema = z.union([z.string(), z.number()]);

const financialSourceSchema = z.object({
  source_id: z.string(),
  metric_name: z.string(),
  reporting_period: z.string(),
  report_period_type: z.enum(["QUARTERLY", "ANNUAL"]),
  statement_basis: z.enum(["CONSOLIDATED", "STANDALONE", "UNKNOWN"]),
  value: decimalSchema,
  comparative_value: decimalSchema.nullable().optional(),
  comparative_period: z.string().nullable().optional(),
  unit: z.string().nullable().optional(),
  currency: z.string().nullable().optional(),
  source_document: z.string(),
  source_page: z.number().int().positive(),
});

const transcriptSourceSchema = z.object({
  source_id: z.string(),
  document_name: z.string(),
  reporting_period: z.string().nullable().optional(),
  page_number: z.number().int().positive(),
  speaker: z.string().nullable().optional(),
  retrieval_score: z.number(),
});

export const sourceReferenceSchema = z.object({
  source: z.union([financialSourceSchema, transcriptSourceSchema]),
  supporting_evidence: z.array(z.string()).default([]),
});

export const assessmentSchema = z.object({
  direction: z.enum(["positive", "stable", "cautious", "negative", "uncertain"]),
  evidence_status: z.enum(["supported", "limited", "conflicting", "unavailable"]),
  outlook: z.string(),
  source_references: z.array(sourceReferenceSchema),
});

const forecastItemSchema = z.object({
  item: z.string(),
  reasoning: z.string(),
  source_references: z.array(sourceReferenceSchema),
});

const forecastReasoningSchema = z.object({
  inference: z.string(),
  source_references: z.array(sourceReferenceSchema),
});

export const quarterlyForecastSchema = z.object({
  company: z.string(),
  forecast_for: z.string(),
  based_on_reporting_periods: z.array(z.string()).min(1),
  overall_business_outlook: assessmentSchema,
  revenue_outlook: assessmentSchema,
  profitability_outlook: assessmentSchema,
  operating_margin_outlook: assessmentSchema,
  management_outlook: assessmentSchema,
  business_risks: z.array(forecastItemSchema),
  business_opportunities: z.array(forecastItemSchema),
  key_growth_drivers: z.array(forecastItemSchema),
  key_monitoring_indicators: z.array(forecastItemSchema),
  confidence_level: z.enum(["high", "medium", "low"]),
  reasoning: z.array(forecastReasoningSchema),
  source_references: z.array(sourceReferenceSchema),
  data_limitations: z.array(z.string()),
});

export const forecastAPIResponseSchema = z.object({
  request_id: z.string().uuid(),
  status: z.literal("completed"),
  forecast: quarterlyForecastSchema,
});

export const healthResponseSchema = z.object({
  status: z.enum(["healthy", "degraded"]),
  application: z.string(),
  database_status: z.enum(["connected", "unavailable", "not_configured"]),
});

export type SourceReference = z.infer<typeof sourceReferenceSchema>;
export type ForecastAssessment = z.infer<typeof assessmentSchema>;
export type QuarterlyForecast = z.infer<typeof quarterlyForecastSchema>;
export type ForecastAPIResponse = z.infer<typeof forecastAPIResponseSchema>;
export type HealthResponse = z.infer<typeof healthResponseSchema>;
