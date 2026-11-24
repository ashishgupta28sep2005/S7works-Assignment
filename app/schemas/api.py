"""HTTP request, response, health, and error schemas."""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.schemas.forecast import QuarterlyForecast


class APIRequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ForecastRequestBody(APIRequestModel):
    company: str = Field(min_length=1, max_length=255, description="Company name supported by the source corpus.")
    question: str = Field(min_length=1, max_length=4000, description="Qualitative forecast request.")

    @field_validator("company", "question")
    @classmethod
    def strip_nonempty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class ForecastAPIResponse(APIRequestModel):
    request_id: UUID
    status: Literal["completed"]
    forecast: QuarterlyForecast


class HealthResponse(APIRequestModel):
    status: Literal["healthy", "degraded"]
    application: str
    database_status: Literal["connected", "unavailable", "not_configured"]


class APIErrorResponse(APIRequestModel):
    error: str
    request_id: UUID | None = None


class ValidationIssue(APIRequestModel):
    field: str
    message: str


class ValidationErrorResponse(APIRequestModel):
    error: str
    issues: list[ValidationIssue]
