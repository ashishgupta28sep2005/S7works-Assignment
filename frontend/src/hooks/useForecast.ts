import { useState } from "react";
import { ApiClientError, requestForecast } from "../services/api";
import type { ForecastAPIResponse } from "../types/api";

export function useForecast() {
  const [response, setResponse] = useState<ForecastAPIResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function generate(company: string, question: string): Promise<boolean> {
    const cleanCompany = company.trim();
    const cleanQuestion = question.trim();
    if (!cleanCompany || !cleanQuestion) {
      setError("Enter both a company and a forecast question.");
      return false;
    }
    if (loading) return false;

    setLoading(true);
    setError(null);
    try {
      const result = await requestForecast(cleanCompany, cleanQuestion);
      setResponse(result);
      return true;
    } catch (caught) {
      setError(caught instanceof ApiClientError ? caught.message : "Forecast request failed. Please try again.");
      return false;
    } finally {
      setLoading(false);
    }
  }

  function clear() {
    setResponse(null);
    setError(null);
  }

  return { response, loading, error, generate, clear, setError };
}
