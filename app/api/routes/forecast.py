"""Versioned forecast and health endpoints backed by the Phase 8 service."""

from __future__ import annotations

import logging
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Request
from sqlalchemy.exc import SQLAlchemyError

from app.agents.forecast_agent import ForecastGenerationError
from app.config.llm import GroqConfigurationError
from app.schemas.api import (
    APIErrorResponse,
    ForecastAPIResponse,
    ForecastRequestBody,
    HealthResponse,
    ValidationErrorResponse,
)
from app.services.forecast_request_service import (
    ForecastRequestService,
    ForecastServiceResult,
    safe_forecast_error_message,
    sanitize_forecast_error,
)

logger = logging.getLogger(__name__)
router = APIRouter()
APP_NAME = "TCS Financial Forecasting Agent"


class APIRequestError(Exception):
    def __init__(self, status_code: int, error: str, request_id: UUID | None = None) -> None:
        self.status_code = status_code
        self.error = error
        self.request_id = request_id
        super().__init__(error)


def get_forecast_request_service(request: Request) -> ForecastRequestService:
    service = getattr(request.app.state, "forecast_request_service", None)
    if service is None:
        raise APIRequestError(503, "Database-backed forecast service is unavailable.")
    return service


def get_health(request: Request) -> HealthResponse:
    engine = getattr(request.app.state, "database_engine", None)
    if engine is None:
        return HealthResponse(
            status="degraded", application=APP_NAME, database_status="not_configured",
        )
    try:
        with engine.connect() as connection:
            connection.exec_driver_sql("SELECT 1")
    except Exception as exc:
        logger.warning("Health database check failed (%s).", type(exc).__name__)
        return HealthResponse(
            status="degraded", application=APP_NAME, database_status="unavailable",
        )
    return HealthResponse(status="healthy", application=APP_NAME, database_status="connected")


@router.get("/health", response_model=HealthResponse, tags=["health"])
def health(request: Request) -> HealthResponse:
    """Report application and database connectivity without exposing details."""
    return get_health(request)


@router.post(
    "/api/v1/forecast",
    response_model=ForecastAPIResponse,
    tags=["forecasts"],
    responses={
        422: {"model": ValidationErrorResponse, "description": "Invalid request body."},
        500: {"model": APIErrorResponse, "description": "Unexpected server error."},
        502: {"model": APIErrorResponse, "description": "Forecast generation failed."},
        503: {"model": APIErrorResponse, "description": "Database or forecast service unavailable."},
    },
)
def create_forecast(
    body: ForecastRequestBody,
    service: Annotated[ForecastRequestService, Depends(get_forecast_request_service)],
) -> ForecastAPIResponse:
    """Run the established Phase 8 service and return its validated forecast."""
    try:
        result: ForecastServiceResult = service.generate_and_record(body.company, body.question)
    except SQLAlchemyError as exc:
        logger.error("Forecast request database operation failed (%s).", type(exc).__name__)
        raise APIRequestError(503, "Database is unavailable; the request could not be recorded.") from None
    except GroqConfigurationError:
        logger.error("Forecast provider configuration is unavailable.")
        raise APIRequestError(503, "Forecast service is not configured.") from None
    except ForecastGenerationError as exc:
        logger.error("Forecast generation failed (%s).", exc.category)
        raise APIRequestError(502, sanitize_forecast_error(exc)) from None
    except Exception as exc:
        logger.error("Forecast request failed unexpectedly (%s).", type(exc).__name__)
        raise APIRequestError(500, "An internal error prevented forecast processing.") from None

    request_id = UUID(str(result.request_id))
    if result.status != "completed" or result.forecast is None:
        raise APIRequestError(
            502,
            safe_forecast_error_message(result.error_message),
            request_id=request_id,
        )
    try:
        return ForecastAPIResponse(
            request_id=request_id,
            status="completed",
            forecast=result.forecast,
        )
    except Exception:
        logger.error("Forecast service returned an invalid response model.")
        raise APIRequestError(500, "An internal error prevented forecast processing.", request_id=request_id) from None
