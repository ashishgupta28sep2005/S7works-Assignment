"""SQLAlchemy ORM models for request and forecast persistence."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utc_now_naive() -> datetime:
    """Return UTC time as a naive value for MySQL DATETIME portability."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass


class ForecastRequest(Base):
    __tablename__ = "forecast_requests"
    __table_args__ = (
        CheckConstraint("status IN ('processing', 'completed', 'failed')", name="ck_forecast_request_status"),
        Index("ix_forecast_requests_timestamp", "request_timestamp"),
        Index("ix_forecast_requests_status", "status"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    request_id: Mapped[str] = mapped_column(String(36), nullable=False, unique=True)
    company_name: Mapped[str] = mapped_column(String(255), nullable=False)
    user_question: Mapped[str] = mapped_column(Text, nullable=False)
    request_timestamp: Mapped[datetime] = mapped_column(DateTime(), nullable=False, default=utc_now_naive)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="processing")
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    responses: Mapped[list["ForecastResponse"]] = relationship(
        back_populates="request", cascade="all, delete-orphan", passive_deletes=True,
    )


class ForecastResponse(Base):
    __tablename__ = "forecast_responses"
    __table_args__ = (
        UniqueConstraint("request_id", name="uq_forecast_responses_request_id"),
        Index("ix_forecast_responses_forecast_for", "forecast_for"),
        Index("ix_forecast_responses_confidence", "confidence_level"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    request_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("forecast_requests.request_id", ondelete="CASCADE"), nullable=False,
    )
    forecast_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    forecast_for: Mapped[str] = mapped_column(String(32), nullable=False)
    confidence_level: Mapped[str] = mapped_column(String(16), nullable=False)
    response_timestamp: Mapped[datetime] = mapped_column(DateTime(), nullable=False, default=utc_now_naive)

    request: Mapped[ForecastRequest] = relationship(back_populates="responses")
