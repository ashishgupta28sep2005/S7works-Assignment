"""FastAPI application for the TCS Financial Forecasting Agent."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import Engine

from app.api.routes.forecast import APIRequestError, APP_NAME, router
from app.config.llm import GroqConfigurationError
from app.database.connection import (
    DatabaseConfigurationError,
    create_database_engine,
    create_session_factory,
)
from app.database.repository import ForecastRepository
from app.config.settings import get_settings
from app.schemas.api import APIErrorResponse, ValidationErrorResponse, ValidationIssue
from app.services.forecast_request_service import ForecastRequestService

logger = logging.getLogger(__name__)
APP_VERSION = "1.0.0"
APP_DESCRIPTION = (
    "Evidence-grounded qualitative TCS forecasts with request and response persistence in MySQL. "
    "The API delegates forecasting and persistence to the existing Phase 8 service."
)


def create_app(
    *,
    database_engine: Engine | None = None,
    forecast_request_service: ForecastRequestService | None = None,
    engine_factory=create_database_engine,
) -> FastAPI:
    """Build the app; optional dependencies allow isolated API tests."""
    externally_configured = database_engine is not None or forecast_request_service is not None

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        owned_engine: Engine | None = None
        if not externally_configured:
            try:
                owned_engine = engine_factory()
                application.state.database_engine = owned_engine
                repository = ForecastRepository(create_session_factory(owned_engine))
                application.state.forecast_request_service = ForecastRequestService(repository)
            except DatabaseConfigurationError:
                application.state.database_engine = None
                application.state.forecast_request_service = None
                logger.warning("Database configuration is incomplete; API is starting in degraded mode.")
            except Exception as exc:
                application.state.database_engine = None
                application.state.forecast_request_service = None
                logger.error("Database runtime setup failed (%s); API is starting in degraded mode.", type(exc).__name__)
        try:
            yield
        finally:
            if owned_engine is not None:
                owned_engine.dispose()

    application = FastAPI(
        title=APP_NAME,
        version=APP_VERSION,
        description=APP_DESCRIPTION,
        lifespan=lifespan,
    )
    application.state.database_engine = database_engine
    application.state.forecast_request_service = forecast_request_service
    application.include_router(router)
    configured_origins = list(get_settings().cors_allowed_origins)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=configured_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Accept", "Content-Type"],
    )

    @application.exception_handler(APIRequestError)
    async def api_request_error_handler(_: Request, exc: APIRequestError) -> JSONResponse:
        payload = APIErrorResponse(error=exc.error, request_id=exc.request_id)
        return JSONResponse(status_code=exc.status_code, content=payload.model_dump(mode="json", exclude_none=True))

    @application.exception_handler(RequestValidationError)
    async def request_validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        # Do not echo rejected request values; validation locations and generic messages are sufficient.
        issues = []
        for error in exc.errors():
            # Only emit known request-property names, never arbitrary keys from the rejected body.
            location = ".".join(
                str(part) for part in error.get("loc", ()) if part in {"company", "question"}
            )
            message = "This field is required." if error.get("type") == "missing" else "Invalid field value."
            issues.append(ValidationIssue(field=location or "request", message=message))
        payload = ValidationErrorResponse(error="Invalid request body.", issues=issues)
        return JSONResponse(status_code=422, content=payload.model_dump(mode="json"))

    @application.exception_handler(Exception)
    async def unexpected_error_handler(_: Request, exc: Exception) -> JSONResponse:
        logger.error("Unhandled API exception (%s).", type(exc).__name__)
        payload = APIErrorResponse(error="An internal server error occurred.")
        return JSONResponse(status_code=500, content=payload.model_dump(mode="json", exclude_none=True))

    return application


app = create_app()
