"""Transactional persistence operations for forecast request records."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker, selectinload

from app.database.models import ForecastRequest, ForecastResponse
from app.schemas.forecast import QuarterlyForecast


class DuplicateRequestID(ValueError):
    """A request identifier already exists."""


class RequestNotFound(LookupError):
    """No request exists for the supplied identifier."""


class ForecastRepository:
    def __init__(self, sessions: sessionmaker[Session]) -> None:
        self._sessions = sessions

    def save_request(
        self,
        request_id: str,
        company_name: str,
        user_question: str,
        *,
        status: str = "processing",
    ) -> ForecastRequest:
        record = ForecastRequest(
            request_id=request_id,
            company_name=company_name,
            user_question=user_question,
            status=status,
        )
        try:
            with self._transaction() as session:
                session.add(record)
                session.flush()
        except IntegrityError as exc:
            if "request_id" in str(exc.orig).casefold() or "unique" in str(exc.orig).casefold():
                raise DuplicateRequestID("A request with this request_id already exists.") from None
            raise
        return record

    def update_request_status(
        self,
        request_id: str,
        status: str,
        *,
        error_message: str | None = None,
    ) -> ForecastRequest:
        if status not in {"processing", "completed", "failed"}:
            raise ValueError("status must be processing, completed, or failed")
        if status == "completed":
            raise ValueError("completed status can only be set when a forecast response is saved")
        if status == "failed" and not error_message:
            raise ValueError("failed requests require a sanitized error_message")
        if status != "failed":
            error_message = None
        with self._transaction() as session:
            record = session.scalar(select(ForecastRequest).where(ForecastRequest.request_id == request_id))
            if record is None:
                raise RequestNotFound("The forecast request does not exist.")
            record.status = status
            record.error_message = error_message
            session.flush()
        return record

    def save_forecast_response(self, request_id: str, forecast: QuarterlyForecast) -> ForecastResponse:
        """Persist the forecast and completed state atomically."""
        if not isinstance(forecast, QuarterlyForecast):
            raise TypeError("forecast must be a validated QuarterlyForecast")
        response = ForecastResponse(
            request_id=request_id,
            forecast_json=forecast.model_dump(mode="json"),
            forecast_for=forecast.forecast_for,
            confidence_level=forecast.confidence_level,
        )
        with self._transaction() as session:
            request = session.scalar(select(ForecastRequest).where(ForecastRequest.request_id == request_id))
            if request is None:
                raise RequestNotFound("The forecast request does not exist.")
            if request.status != "processing":
                raise ValueError("Only a processing request can be completed.")
            session.add(response)
            request.status = "completed"
            request.error_message = None
            session.flush()
        return response

    def get_request(self, request_id: str) -> ForecastRequest | None:
        with self._session() as session:
            return session.scalar(
                select(ForecastRequest)
                .options(selectinload(ForecastRequest.responses))
                .where(ForecastRequest.request_id == request_id)
            )

    @contextmanager
    def _transaction(self) -> Iterator[Session]:
        """Yield one transaction and explicitly roll it back on all failures."""
        session = self._sessions()
        try:
            with session.begin():
                yield session
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    @contextmanager
    def _session(self) -> Iterator[Session]:
        session = self._sessions()
        try:
            yield session
        finally:
            session.close()

