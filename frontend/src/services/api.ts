import {
  forecastAPIResponseSchema,
  healthResponseSchema,
  type ForecastAPIResponse,
  type HealthResponse,
} from "../types/api";

export const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL?.trim() || "http://127.0.0.1:8000"
).replace(/\/+$/, "");

export type ApiErrorKind = "http" | "network" | "timeout" | "invalid_response";

export class ApiClientError extends Error {
  readonly kind: ApiErrorKind;
  readonly status?: number;

  constructor(message: string, kind: ApiErrorKind, status?: number) {
    super(message);
    this.name = "ApiClientError";
    this.kind = kind;
    this.status = status;
  }
}

async function fetchWithTimeout(url: string, init: RequestInit = {}, timeoutMs = 120_000): Promise<Response> {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...init, signal: controller.signal });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      throw new ApiClientError("The request timed out. Please try again.", "timeout");
    }
    throw new ApiClientError(
      "Could not reach the forecasting service. Check that the backend is running and try again.",
      "network",
    );
  } finally {
    window.clearTimeout(timer);
  }
}

async function jsonResponse(response: Response): Promise<unknown> {
  try {
    return await response.json();
  } catch {
    throw new ApiClientError("The backend returned an unreadable response.", "invalid_response", response.status);
  }
}

function backendErrorMessage(payload: unknown, status: number): string {
  if (payload && typeof payload === "object" && "error" in payload && typeof payload.error === "string") {
    return payload.error;
  }
  if (status === 422) return "Please check the company and question fields.";
  if (status === 503) return "The database-backed forecast service is temporarily unavailable.";
  if (status >= 500) return "The server could not complete the request. Please try again later.";
  return "The request could not be completed.";
}

export async function fetchHealth(): Promise<HealthResponse> {
  const response = await fetchWithTimeout(`${API_BASE_URL}/health`, {}, 8_000);
  if (!response.ok) {
    throw new ApiClientError("The backend health check failed.", "http", response.status);
  }
  const parsed = healthResponseSchema.safeParse(await jsonResponse(response));
  if (!parsed.success) {
    throw new ApiClientError("The backend returned an invalid health response.", "invalid_response", response.status);
  }
  return parsed.data;
}

export async function requestForecast(company: string, question: string): Promise<ForecastAPIResponse> {
  const response = await fetchWithTimeout(`${API_BASE_URL}/api/v1/forecast`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify({ company, question }),
  });
  const payload = await jsonResponse(response);
  if (!response.ok) {
    throw new ApiClientError(backendErrorMessage(payload, response.status), "http", response.status);
  }
  const parsed = forecastAPIResponseSchema.safeParse(payload);
  if (!parsed.success) {
    throw new ApiClientError("The backend returned a forecast that did not match its published schema.", "invalid_response", response.status);
  }
  return parsed.data;
}
