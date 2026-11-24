"""Callable service and CLI for recording forecast requests and responses."""

from __future__ import annotations

import argparse
import logging
import uuid
from dataclasses import dataclass
from typing import Any

from app.agents.forecast_agent import ForecastAgent, ForecastGenerationError, ForecastPeriodError
from app.config.llm import GroqConfigurationError
from app.database.connection import (
    DatabaseConfigurationError,
    create_database_engine,
    create_session_factory,
)
from app.database.repository import ForecastRepository
from app.schemas.forecast import QuarterlyForecast

logger = logging.getLogger(__name__)

_SAFE_FORECAST_ERRORS = {
    "authentication": "Forecast provider authentication failed.",
    "rate_limit": "Forecast provider rate limit reached; retry later.",
    "timeout": "Forecast provider request timed out.",
    "invalid_response": "Forecast provider returned an invalid structured response.",
    "request_rejected": "Forecast provider rejected its structured-output request; check the backend log for HTTP status.",
    "context_limit": "Forecast provider rejected the request because it exceeded its context limit.",
    "provider_error": "Forecast provider could not complete the request; check the backend log for its HTTP status.",
    "invalid_source_reference": "Forecast was rejected because its source references were invalid.",
    "unsupported_numeric_forecast": "Forecast was rejected because it contained an unsupported numeric claim.",
    "insufficient_evidence": "Forecast could not be generated from the available evidence.",
}
_GENERIC_FORECAST_ERROR = "Forecast generation failed; details were withheld."


@dataclass(frozen=True)
class ForecastServiceResult:
    request_id: str
    status: str
    forecast: QuarterlyForecast | None = None
    error_message: str | None = None


def sanitize_forecast_error(exc: Exception) -> str:
    """Return a fixed safe message; arbitrary provider exception text is discarded."""
    if isinstance(exc, ForecastGenerationError):
        return _SAFE_FORECAST_ERRORS.get(exc.category, _GENERIC_FORECAST_ERROR)
    if isinstance(exc, GroqConfigurationError):
        return "Forecast service configuration is incomplete."
    if isinstance(exc, ForecastPeriodError):
        return "Forecast period could not be determined from the available reporting data."
    return _GENERIC_FORECAST_ERROR


def safe_forecast_error_message(message: str | None) -> str:
    """Allow only fixed, credential-safe messages through the HTTP boundary."""
    if message in _SAFE_FORECAST_ERRORS.values():
        return message
    return _GENERIC_FORECAST_ERROR


class ForecastRequestService:
    """Persist a request, run Phase 7, and atomically save its validated response."""

    def __init__(
        self,
        repository: ForecastRepository,
        forecast_agent: Any | None = None,
    ) -> None:
        self.repository = repository
        self.forecast_agent = forecast_agent

    def generate_and_record(self, company_name: str, user_question: str) -> ForecastServiceResult:
        company = company_name.strip()
        question = user_question.strip()
        if not company:
            raise ValueError("company_name cannot be empty")
        if not question:
            raise ValueError("user_question cannot be empty")

        request_id = str(uuid.uuid4())
        self.repository.save_request(request_id, company, question)
        try:
            agent = self.forecast_agent if self.forecast_agent is not None else ForecastAgent()
            forecast = agent.generate(company, question)
            if not isinstance(forecast, QuarterlyForecast):
                raise TypeError("ForecastAgent returned an unvalidated forecast.")
            self.repository.save_forecast_response(request_id, forecast)
        except Exception as exc:
            safe_message = sanitize_forecast_error(exc)
            logger.warning("Forecast request %s failed (%s).", request_id, type(exc).__name__)
            try:
                self.repository.update_request_status(
                    request_id, "failed", error_message=safe_message,
                )
            except Exception as persistence_error:
                logger.error(
                    "Could not persist failed status for request %s (%s).",
                    request_id,
                    type(persistence_error).__name__,
                )
                return ForecastServiceResult(
                    request_id=request_id,
                    status="failed",
                    error_message="Forecast failed and its failure status could not be persisted.",
                )
            return ForecastServiceResult(request_id, "failed", error_message=safe_message)
        return ForecastServiceResult(request_id, "completed", forecast=forecast)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a TCS forecast and record it in MySQL.")
    parser.add_argument("--company", default="TCS", help="Company name (the current corpus is TCS-only).")
    parser.add_argument("--question", required=True, help="Forecast request to record.")
    args = parser.parse_args(argv)
    try:
        engine = create_database_engine()
    except DatabaseConfigurationError as exc:
        print(f"Database configuration error: {exc}")
        return 2

    try:
        repository = ForecastRepository(create_session_factory(engine))
        result = ForecastRequestService(repository).generate_and_record(args.company, args.question)
    except Exception as exc:
        logger.error("Forecast request service failed (%s); sensitive details were withheld.", type(exc).__name__)
        print(f"Request service failed ({type(exc).__name__}); sensitive details were withheld.")
        return 1
    finally:
        engine.dispose()

    print(f"request_id: {result.request_id}")
    print(f"status: {result.status}")
    if result.forecast is not None:
        print(result.forecast.model_dump_json(indent=2, exclude_none=True))
        return 0
    print(f"error: {result.error_message}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
