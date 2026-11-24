"""FastAPI unit tests: all forecasting and database work is injected or mocked."""

from __future__ import annotations

from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

from app.agents.forecast_agent import ForecastGenerationError
from app.database.connection import create_database_engine
from app.main import APP_NAME, create_app
from app.schemas.api import ForecastAPIResponse, ForecastRequestBody
from app.schemas.forecast import ForecastAssessment, QuarterlyForecast
from app.services.forecast_request_service import ForecastServiceResult


def make_forecast() -> QuarterlyForecast:
    unknown = ForecastAssessment(
        direction="uncertain",
        evidence_status="unavailable",
        outlook="Insufficient evidence is available for this outlook.",
        source_references=[],
    )
    return QuarterlyForecast(
        company="TCS",
        forecast_for="Q3 FY27",
        based_on_reporting_periods=["Q2 FY27"],
        overall_business_outlook=unknown,
        revenue_outlook=unknown,
        profitability_outlook=unknown,
        operating_margin_outlook=unknown,
        management_outlook=unknown,
        confidence_level="low",
        data_limitations=["API unit-test fixture."],
    )


class MockForecastRequestService:
    def __init__(self, result=None, error: Exception | None = None):
        self.result = result
        self.error = error
        self.calls: list[tuple[str, str]] = []

    def generate_and_record(self, company: str, question: str):
        self.calls.append((company, question))
        if self.error:
            raise self.error
        return self.result


def make_client(*, result=None, error=None, engine=None):
    request_service = MockForecastRequestService(result=result, error=error)
    database_engine = engine or create_database_engine(database_url="sqlite+pysqlite:///:memory:")
    app = create_app(database_engine=database_engine, forecast_request_service=request_service)
    return TestClient(app), request_service, database_engine


def test_health_reports_application_and_connected_database():
    client, _, engine = make_client()
    try:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {
            "status": "healthy",
            "application": APP_NAME,
            "database_status": "connected",
        }
    finally:
        engine.dispose()


class UnavailableEngine:
    def connect(self):
        raise OperationalError("SELECT 1", {}, RuntimeError("MYSQL_PASSWORD=hidden-db-secret"))


def test_health_reports_unavailable_database_without_internal_details():
    client, _, _ = make_client(engine=UnavailableEngine())
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["database_status"] == "unavailable"
    assert response.json()["status"] == "degraded"
    assert "hidden-db-secret" not in response.text


def test_health_reports_not_configured_without_database_engine():
    def missing_configuration():
        from app.database.connection import DatabaseConfigurationError
        raise DatabaseConfigurationError("Missing database configuration: MYSQL_HOST.")

    app = create_app(engine_factory=missing_configuration)
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["database_status"] == "not_configured"
        assert response.json()["status"] == "degraded"


def test_valid_forecast_request_uses_service_and_returns_request_id():
    request_id = uuid4()
    client, service, engine = make_client(
        result=ForecastServiceResult(request_id=str(request_id), status="completed", forecast=make_forecast()),
    )
    try:
        response = client.post("/api/v1/forecast", json={
            "company": " TCS ",
            "question": " Give a qualitative outlook for the upcoming quarter ",
        })
        assert response.status_code == 200
        payload = ForecastAPIResponse.model_validate(response.json())
        assert payload.request_id == request_id
        assert payload.status == "completed"
        assert payload.forecast.forecast_for == "Q3 FY27"
        assert service.calls == [("TCS", "Give a qualitative outlook for the upcoming quarter")]
    finally:
        engine.dispose()


@pytest.mark.parametrize("body", [
    {"question": "Forecast"},
    {"company": "TCS"},
    {"company": " ", "question": "Forecast"},
    {"company": "TCS", "question": " "},
])
def test_invalid_or_missing_request_fields_return_422_without_echo(body):
    client, _, engine = make_client()
    try:
        response = client.post("/api/v1/forecast", json=body)
        assert response.status_code == 422
        assert response.json()["error"] == "Invalid request body."
        assert "input" not in response.text.casefold()
    finally:
        engine.dispose()


def test_extra_request_fields_are_rejected():
    client, _, engine = make_client()
    try:
        response = client.post("/api/v1/forecast", json={
            "company": "TCS", "question": "Forecast", "api_key": "do-not-echo",
        })
        assert response.status_code == 422
        assert "do-not-echo" not in response.text
    finally:
        engine.dispose()


def test_rejected_body_does_not_echo_untrusted_field_names():
    client, _, engine = make_client()
    try:
        response = client.post("/api/v1/forecast", json={
            "company": "TCS", "question": "Forecast", "MYSQL_PASSWORD=hidden-db-secret": "value",
        })
        assert response.status_code == 422
        assert "hidden-db-secret" not in response.text
    finally:
        engine.dispose()


def test_service_failure_returns_safe_502_and_request_id():
    request_id = uuid4()
    client, _, engine = make_client(result=ForecastServiceResult(
        request_id=str(request_id), status="failed", error_message="GROQ_API_KEY=hidden-secret",
    ))
    try:
        response = client.post("/api/v1/forecast", json={"company": "TCS", "question": "Forecast"})
        assert response.status_code == 502
        assert response.json() == {
            "error": "Forecast generation failed; details were withheld.",
            "request_id": str(request_id),
        }
        assert "hidden-secret" not in response.text
    finally:
        engine.dispose()


def test_raised_forecast_failure_returns_safe_502():
    client, _, engine = make_client(error=ForecastGenerationError(
        "raw provider error with password=secret", category="authentication",
    ))
    try:
        response = client.post("/api/v1/forecast", json={"company": "TCS", "question": "Forecast"})
        assert response.status_code == 502
        assert response.json()["error"] == "Forecast provider authentication failed."
        assert "secret" not in response.text
    finally:
        engine.dispose()


def test_service_failure_exposes_only_allowlisted_safe_error_message():
    client, _, engine = make_client(result=ForecastServiceResult(
        request_id=str(uuid4()), status="failed",
        error_message="Forecast provider rate limit reached; retry later.",
    ))
    try:
        response = client.post("/api/v1/forecast", json={"company": "TCS", "question": "Forecast"})
        assert response.status_code == 502
        assert response.json()["error"] == "Forecast provider rate limit reached; retry later."
    finally:
        engine.dispose()


def test_database_unavailable_returns_safe_503():
    client, _, engine = make_client(error=OperationalError(
        "INSERT", {}, RuntimeError("MYSQL_PASSWORD=hidden-db-secret"),
    ))
    try:
        response = client.post("/api/v1/forecast", json={"company": "TCS", "question": "Forecast"})
        assert response.status_code == 503
        assert "hidden-db-secret" not in response.text
        assert response.json()["error"] == "Database is unavailable; the request could not be recorded."
    finally:
        engine.dispose()


def test_unexpected_error_returns_generic_500():
    client, _, engine = make_client(error=RuntimeError("internal key=private"))
    try:
        response = client.post("/api/v1/forecast", json={"company": "TCS", "question": "Forecast"})
        assert response.status_code == 500
        assert response.json()["error"] == "An internal error prevented forecast processing."
        assert "private" not in response.text
    finally:
        engine.dispose()


def test_response_schema_requires_valid_forecast_model():
    valid = {
        "request_id": str(uuid4()),
        "status": "completed",
        "forecast": make_forecast().model_dump(mode="json"),
    }
    assert ForecastAPIResponse.model_validate(valid).forecast.forecast_for == "Q3 FY27"
    valid["forecast"] = {"forecast_for": "Q3 FY27"}
    with pytest.raises(Exception):
        ForecastAPIResponse.model_validate(valid)


def test_openapi_and_documentation_routes_include_forecast_contract():
    client, _, engine = make_client()
    try:
        assert client.get("/docs").status_code == 200
        assert client.get("/redoc").status_code == 200
        schema_response = client.get("/openapi.json")
        assert schema_response.status_code == 200
        schema = schema_response.json()
        assert "/api/v1/forecast" in schema["paths"]
        assert "ForecastRequestBody" in schema["components"]["schemas"]
        assert "QuarterlyForecast" in schema["components"]["schemas"]
    finally:
        engine.dispose()


def test_cors_allows_local_vite_origin_but_not_arbitrary_origins():
    client, _, engine = make_client()
    try:
        allowed = client.options(
            "/api/v1/forecast",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
        denied = client.options(
            "/api/v1/forecast",
            headers={
                "Origin": "https://untrusted.example",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert allowed.headers.get("access-control-allow-origin") == "http://localhost:5173"
        assert denied.headers.get("access-control-allow-origin") is None
    finally:
        engine.dispose()
