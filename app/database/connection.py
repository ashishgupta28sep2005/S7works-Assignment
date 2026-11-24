"""Database configuration and SQLAlchemy engine/session construction."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from dotenv import load_dotenv
from sqlalchemy import Engine, create_engine as sqlalchemy_create_engine
from sqlalchemy.engine import URL
from sqlalchemy.orm import Session, sessionmaker

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATABASE_NAME = "tcs_financial_forecasting"


class DatabaseConfigurationError(ValueError):
    """Database configuration is incomplete or invalid, without echoing values."""


@dataclass(frozen=True)
class DatabaseSettings:
    host: str
    port: int
    username: str
    password: str = field(repr=False)
    database: str = DATABASE_NAME

    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str] | None = None,
        *,
        load_env_file: bool = True,
    ) -> "DatabaseSettings":
        if environ is None:
            if load_env_file:
                load_dotenv(PROJECT_ROOT / ".env", override=False)
            values: Mapping[str, str] = os.environ
        else:
            values = environ

        required = ("MYSQL_HOST", "MYSQL_PORT", "MYSQL_USER", "MYSQL_PASSWORD", "MYSQL_DATABASE")
        missing = [key for key in required if key not in values]
        if missing:
            raise DatabaseConfigurationError(
                "Missing database configuration: " + ", ".join(missing)
                + ". Set these variables in .env or the process environment."
            )

        try:
            port = int(values["MYSQL_PORT"])
        except (TypeError, ValueError):
            raise DatabaseConfigurationError("MYSQL_PORT must be an integer between 1 and 65535.") from None
        if not 1 <= port <= 65535:
            raise DatabaseConfigurationError("MYSQL_PORT must be an integer between 1 and 65535.")

        host = values["MYSQL_HOST"].strip()
        username = values["MYSQL_USER"].strip()
        database = values["MYSQL_DATABASE"].strip()
        if not host or not username or not database:
            raise DatabaseConfigurationError("MYSQL_HOST, MYSQL_USER, and MYSQL_DATABASE cannot be empty.")
        if database != DATABASE_NAME:
            raise DatabaseConfigurationError(f"MYSQL_DATABASE must be {DATABASE_NAME!r} for this project.")
        return cls(
            host=host,
            port=port,
            username=username,
            password=values["MYSQL_PASSWORD"],
            database=database,
        )

    def url(self) -> URL:
        """Build a safely escaped SQLAlchemy URL; its repr masks the password."""
        return URL.create(
            "mysql+pymysql",
            username=self.username,
            password=self.password,
            host=self.host,
            port=self.port,
            database=self.database,
            query={"charset": "utf8mb4"},
        )


def create_database_engine(
    settings: DatabaseSettings | None = None,
    *,
    database_url: str | URL | None = None,
    **engine_options: object,
) -> Engine:
    """Create a pre-ping engine, optionally accepting a test-only database URL."""
    if settings is not None and database_url is not None:
        raise ValueError("Provide either database settings or a database URL, not both.")
    target = database_url if database_url is not None else (settings or DatabaseSettings.from_environment()).url()
    options = {"pool_pre_ping": True, **engine_options}
    return sqlalchemy_create_engine(target, **options)


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
