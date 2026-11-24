"""SQLite-backed Phase 8 tests; only the opt-in integration test contacts MySQL."""

from __future__ import annotations

import os
from uuid import uuid4

import pytest
from sqlalchemy import inspect, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.pool import StaticPool
from sqlalchemy.orm import sessionmaker

from app.agents.forecast_agent import ForecastGenerationError
from app.database.connection import (
    DATABASE_NAME,
    DatabaseConfigurationError,
    DatabaseSettings,
    create_database_engine,
    create_session_factory,
)
from app.database.initialize import initialize_database
from app.database.models import Base, ForecastRequest, ForecastResponse
from app.database.repository import DuplicateRequestID, ForecastRepository
from app.schemas.forecast import ForecastAssessment, QuarterlyForecast
from app.services.forecast_request_service import (
    ForecastRequestService,
    sanitize_forecast_error,
)


@pytest.fixture
def db():
    engine = create_database_engine(
        database_url="sqlite+pysqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    repo = ForecastRepository(create_session_factory(engine))
    yield engine, repo
    engine.dispose()


def make_forecast() -> QuarterlyForecast:
    unavailable = ForecastAssessment(
        direction="uncertain",
        evidence_status="unavailable",
        outlook="Insufficient evidence is available for this outlook.",
        source_references=[],
    )
    return QuarterlyForecast(
        company="TCS",
        forecast_for="Q3 FY27",
        based_on_reporting_periods=["Q2 FY27"],
        overall_business_outlook=unavailable,
        revenue_outlook=unavailable,
        profitability_outlook=unavailable,
        operating_margin_outlook=unavailable,
        management_outlook=unavailable,
        confidence_level="low",
        data_limitations=["Technical database test fixture."],
    )


def test_database_configuration_requires_all_variables_without_values_in_error():
    with pytest.raises(DatabaseConfigurationError) as error:
        DatabaseSettings.from_environment({}, load_env_file=False)
    assert "MYSQL_HOST" in str(error.value)
    assert "MYSQL_PASSWORD" in str(error.value)
    assert "tcs_financial_forecasting" not in str(error.value)


def test_database_configuration_builds_hidden_password_url():
    settings = DatabaseSettings.from_environment({
        "MYSQL_HOST": "localhost",
        "MYSQL_PORT": "3306",
        "MYSQL_USER": "test_user",
        "MYSQL_PASSWORD": "local-secret-value",
        "MYSQL_DATABASE": DATABASE_NAME,
    }, load_env_file=False)
    assert settings.url().drivername == "mysql+pymysql"
    assert settings.url().database == DATABASE_NAME
    assert settings.url().render_as_string(hide_password=True).find("local-secret-value") == -1
    assert "local-secret-value" not in repr(settings)


def test_invalid_database_port_is_rejected():
    env = {
        "MYSQL_HOST": "localhost", "MYSQL_PORT": "70000", "MYSQL_USER": "user",
        "MYSQL_PASSWORD": "", "MYSQL_DATABASE": DATABASE_NAME,
    }
    with pytest.raises(DatabaseConfigurationError, match="MYSQL_PORT"):
        DatabaseSettings.from_environment(env, load_env_file=False)


def test_sqlite_connection_and_safe_table_initialization(db):
    engine, _ = db
    with engine.connect() as connection:
        assert connection.execute(text("SELECT 1")).scalar_one() == 1
    initialize_database(engine)
    names = set(inspect(engine).get_table_names())
    assert names == {"forecast_requests", "forecast_responses"}
    assert "request_id" in {column["name"] for column in inspect(engine).get_columns("forecast_requests")}


def test_save_request_and_forecast_json_are_linked(db):
    _, repository = db
    repository.save_request("request-1", "TCS", "Upcoming quarter outlook")
    forecast = make_forecast()
    response = repository.save_forecast_response("request-1", forecast)
    loaded = repository.get_request("request-1")
    assert loaded is not None
    assert loaded.status == "completed"
    assert loaded.error_message is None
    assert len(loaded.responses) == 1
    assert loaded.responses[0].request_id == response.request_id
    assert loaded.responses[0].forecast_json["forecast_for"] == "Q3 FY27"
    assert loaded.responses[0].forecast_json["confidence_level"] == "low"


def test_duplicate_request_id_is_rejected(db):
    _, repository = db
    repository.save_request("same-id", "TCS", "first")
    with pytest.raises(DuplicateRequestID):
        repository.save_request("same-id", "TCS", "second")
    loaded = repository.get_request("same-id")
    assert loaded is not None
    assert loaded.user_question == "first"


def test_response_write_failure_rolls_back_request_status(db):
    engine, repository = db
    repository.save_request("rollback-id", "TCS", "test rollback")
    sessions = create_session_factory(engine)
    with sessions.begin() as session:
        session.add(ForecastResponse(
            request_id="rollback-id", forecast_json={}, forecast_for="Q3 FY27", confidence_level="low",
        ))
    with pytest.raises(IntegrityError):
        repository.save_forecast_response("rollback-id", make_forecast())
    loaded = repository.get_request("rollback-id")
    assert loaded is not None
    assert loaded.status == "processing"
    assert len(loaded.responses) == 1


def test_failed_status_requires_an_error_and_clears_error_when_retried(db):
    _, repository = db
    repository.save_request("status-id", "TCS", "status test")
    with pytest.raises(ValueError, match="only be set when a forecast response is saved"):
        repository.update_request_status("status-id", "completed")
    with pytest.raises(ValueError, match="sanitized error_message"):
        repository.update_request_status("status-id", "failed")
    repository.update_request_status("status-id", "failed", error_message="safe failure")
    repository.update_request_status("status-id", "processing")
    loaded = repository.get_request("status-id")
    assert loaded is not None and loaded.status == "processing" and loaded.error_message is None


class SuccessfulAgent:
    def generate(self, company: str, question: str) -> QuarterlyForecast:
        return make_forecast().model_copy(update={"company": company})


class FailingAgent:
    def generate(self, company: str, question: str) -> QuarterlyForecast:
        raise ForecastGenerationError("provider response had secret local token", category="authentication")


def test_service_generates_uuid_and_persists_validated_forecast(db):
    _, repository = db
    result = ForecastRequestService(repository, SuccessfulAgent()).generate_and_record("TCS", "Forecast")
    assert result.status == "completed"
    assert result.forecast is not None
    assert len(result.request_id) == 36
    record = repository.get_request(result.request_id)
    assert record is not None and record.status == "completed"
    assert len(record.responses) == 1


def test_service_records_failure_with_sanitized_error(db):
    _, repository = db
    result = ForecastRequestService(repository, FailingAgent()).generate_and_record("TCS", "Forecast")
    assert result.status == "failed"
    assert result.error_message == "Forecast provider authentication failed."
    assert "secret" not in result.error_message.casefold()
    record = repository.get_request(result.request_id)
    assert record is not None and record.status == "failed"
    assert record.responses == []
    assert "secret" not in (record.error_message or "").casefold()


def test_unknown_failure_text_is_never_persisted():
    safe = sanitize_forecast_error(RuntimeError("MYSQL_PASSWORD=hunter2 GROQ_API_KEY=secret"))
    assert safe == "Forecast generation failed; details were withheld."
    assert "hunter2" not in safe
    assert "secret" not in safe


@pytest.mark.skipif(
    os.getenv("RUN_MYSQL_INTEGRATION_TESTS", "false").casefold() != "true",
    reason="Set RUN_MYSQL_INTEGRATION_TESTS=true to use the configured local MySQL database.",
)
def test_optional_mysql_connection_and_tables():
    try:
        settings = DatabaseSettings.from_environment()
    except DatabaseConfigurationError as exc:
        pytest.skip(str(exc))
    engine = create_database_engine(settings)
    try:
        initialize_database(engine)
        with engine.connect() as connection:
            assert connection.execute(select(1)).scalar_one() == 1
        assert {"forecast_requests", "forecast_responses"}.issubset(inspect(engine).get_table_names())

        # Use one outer transaction plus repository savepoints so this real-MySQL
        # linkage check leaves no request rows behind after it finishes.
        request_id = str(uuid4())
        with engine.connect() as connection:
            transaction = connection.begin()
            try:
                sessions = sessionmaker(
                    bind=connection,
                    autoflush=False,
                    expire_on_commit=False,
                    join_transaction_mode="create_savepoint",
                )
                repository = ForecastRepository(sessions)
                repository.save_request(request_id, "Phase 9 integration test", "Rollback-scoped test request")
                repository.save_forecast_response(request_id, make_forecast())
                linked = repository.get_request(request_id)
                assert linked is not None and linked.status == "completed"
                assert len(linked.responses) == 1
                assert linked.responses[0].request_id == request_id
                transaction.rollback()
            except Exception:
                transaction.rollback()
                raise
    finally:
        engine.dispose()
