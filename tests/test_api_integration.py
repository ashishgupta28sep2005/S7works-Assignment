"""Explicit opt-in test for the complete MySQL + Groq-backed HTTP path."""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from app.config.settings import get_settings
from app.database.connection import DatabaseConfigurationError, DatabaseSettings
from app.main import create_app


@pytest.mark.skipif(
    os.getenv("RUN_API_INTEGRATION_TESTS", "false").casefold() != "true",
    reason="Set RUN_API_INTEGRATION_TESTS=true for a real MySQL write and Groq forecast call.",
)
def test_real_mysql_and_groq_forecast_api():
    try:
        DatabaseSettings.from_environment()
    except DatabaseConfigurationError as exc:
        pytest.skip(str(exc))
    if not get_settings().groq_api_key:
        pytest.skip("A Groq key is not configured in the local environment.")

    app = create_app()
    with TestClient(app) as client:
        health = client.get("/health")
        if health.json().get("database_status") != "connected":
            pytest.skip("The configured MySQL database is not reachable.")
        response = client.post("/api/v1/forecast", json={
            "company": "TCS",
            "question": "Give a qualitative outlook for the upcoming quarter",
        })
        assert response.status_code == 200, response.json().get("error", "API request failed")
        assert response.json()["request_id"]
        assert response.json()["forecast"]["forecast_for"]
