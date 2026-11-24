"""Validated metadata models for transcript chunks and persisted vector snapshots."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator


class TranscriptChunk(BaseModel):
    company: str
    document_name: str
    document_type: str = "EARNINGS_TRANSCRIPT"
    reporting_period: str | None = None
    earnings_call_date: str | None = None
    page_number: int = Field(ge=1)
    speaker: str | None = None
    speaker_role: Literal["management", "analyst", "moderator"] | None = None
    chunk_id: str
    original_text: str = Field(min_length=1)
    document_hash: str = Field(min_length=16)

    @field_validator("document_type")
    @classmethod
    def transcript_only(cls, value: str) -> str:
        if value != "EARNINGS_TRANSCRIPT":
            raise ValueError("Only earnings transcript chunks may be indexed")
        return value


class IndexSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    generation: str
    embedding_model: str
    dimension: int = Field(gt=0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    chunks: list[TranscriptChunk]
    document_hashes: dict[str, list[str]]


class CurrentIndexPointer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    generation: str
    index_sha256: str
    metadata_sha256: str

    @field_validator("generation")
    @classmethod
    def safe_generation(cls, value: str) -> str:
        if not value.isalnum() or len(value) != 32:
            raise ValueError("Invalid vector-store generation identifier")
        return value


class RetrievalHit(BaseModel):
    score: float
    relevant: bool
    chunk: TranscriptChunk


class RetrievalResponse(BaseModel):
    query: str
    matches: list[RetrievalHit] = Field(default_factory=list)
    rejected_reason: str | None = None
